import gzip
import base64
import logging
import random
import os
import sys

import numpy as np
import xml.etree.ElementTree as ET
from scipy.ndimage import distance_transform_cdt


def resource_path(relative_path):
    """Resolve path to bundled data file (PyInstaller-compatible)."""
    base = getattr(sys, '_MEIPASS', os.path.abspath('.'))
    return os.path.join(base, relative_path)

from wizard_state import WizardState, WizardStep
from procedural_map_generator_functions import (
    subdivide,
    randomize,
    mirror,
    scale_matrix,
    perlin,
    generate_level,
    add_resource_pulls,
    add_command_centers,
    smooth_terrain_tiles,
    add_decoration_tiles,
    place_resource_pull,
    get_neighbors,
    get_all_neighbors,
    DECORATION_FREQUENCY,
    _get_mirrored_positions,
    _is_valid_pool_position,
    _find_valid_resource_positions,
    _get_forbidden_zones,
    _find_mirror_axis_positions,
)

logger = logging.getLogger(__name__)

TILE_ID_OFFSET = 201

# large-rock tileset: firstgid=336, so local ID offset from AutoLight = 135
# Storing (135 + rock_local_id) in id_matrix means encoding gives 201 + 135 + local = 336 + local
ROCK_ID_OFFSET = 135

# large-rock tile local IDs mapped to the same layout as terrain tile_sets:
# (flat_below, center, NW, N, NE, W, E, SW, S, SE, iTL, iTR, iBL, iBR)
LARGE_ROCK_TILE_SET = (-1, 4, 0, 1, 2, 3, 5, 6, 7, 8, 13, 12, 10, 9)

# Cardinal neighbor pattern -> tile_set index (same as _CARDINAL_TILE_MAP in proc functions)
# Neighbors: (top, right, bottom, left), 1 = passable (not wall)
_WALL_CARDINAL_MAP = {
    (0, 0, 0, 0): 0,   # all walls -> center
    (1, 0, 0, 0): 2,   # top passable -> N edge
    (0, 1, 0, 0): 5,   # right passable -> E edge
    (1, 1, 0, 0): 3,   # top+right -> NE corner
    (0, 0, 1, 0): 7,   # bottom passable -> S edge
    (0, 1, 1, 0): 8,   # right+bottom -> SE corner
    (0, 0, 0, 1): 4,   # left passable -> W edge
    (1, 0, 0, 1): 1,   # top+left -> NW corner
    (0, 0, 1, 1): 6,   # bottom+left -> SW corner
}

# Patterns that create thin/isolated walls (same as terrain) - demote to passable
_WALL_ISOLATED = [
    (1, 0, 1, 0), (1, 1, 1, 0), (0, 1, 0, 1),
    (1, 1, 0, 1), (1, 0, 1, 1), (0, 1, 1, 1), (1, 1, 1, 1),
]


def smooth_wall_tiles(wall_matrix):
    """Smooth wall_matrix cells into large-rock tile GIDs for a separate Walls layer.

    For each wall cell (value 1), determines the correct rock tile based on cardinal
    neighbors, then checks diagonal neighbors for inner corners.  Gap cells (value 2)
    are treated as wall-like for neighbour detection but do NOT generate tiles, so they
    create passable openings in the wall border.

    Returns a wall_id_matrix where non-zero cells contain the full GID (firstgid + local_id)
    for the large-rock tileset. Zero means no wall tile at that position.
    """
    if wall_matrix is None:
        return None

    h, w = wall_matrix.shape
    tile_set = LARGE_ROCK_TILE_SET
    # GIDs: firstgid(336) = TILE_ID_OFFSET(201) + ROCK_ID_OFFSET(135)
    ROCK_FIRSTGID = TILE_ID_OFFSET + ROCK_ID_OFFSET
    result = np.zeros((h, w), dtype=int)

    # Helper: a cell is "solid" (wall-like) if it is wall (1) or gap (2).
    # Only empty (0) / out-of-bounds counts as passable for tile selection.
    def _passable(r, c, default):
        """Return 1 if cell at (r,c) is passable (empty), 0 if solid. *default* used for OOB."""
        if r < 0 or r >= h or c < 0 or c >= w:
            return default
        return int(cleaned[r, c] == 0)

    # First pass: remove isolated wall cells (thin lines)
    cleaned = wall_matrix.copy()
    for row in range(h):
        for col in range(w):
            if cleaned[row, col] != 1:
                continue
            top = _passable(row - 1, col, 1)
            right = _passable(row, col + 1, 1)
            bottom = _passable(row + 1, col, 1)
            left = _passable(row, col - 1, 1)
            pattern = (top, right, bottom, left)
            if pattern in _WALL_ISOLATED:
                cleaned[row, col] = 0

    # Second pass: assign tile IDs (only for wall cells, not gap cells)
    for row in range(h):
        for col in range(w):
            if cleaned[row, col] != 1:
                continue

            top = _passable(row - 1, col, 0)
            right = _passable(row, col + 1, 0)
            bottom = _passable(row + 1, col, 0)
            left = _passable(row, col - 1, 0)
            pattern = (top, right, bottom, left)

            if pattern in _WALL_CARDINAL_MAP:
                idx = _WALL_CARDINAL_MAP[pattern]
                if idx == 0:
                    # All cardinal neighbors are solid — check diagonals for inner corners
                    tl = _passable(row - 1, col - 1, 0)
                    tr = _passable(row - 1, col + 1, 0)
                    bl = _passable(row + 1, col - 1, 0)
                    br = _passable(row + 1, col + 1, 0)

                    # Pick inner corner tile if exactly one diagonal is passable
                    if tl and not tr and not bl and not br:
                        local_id = tile_set[10]  # inner top-left
                    elif tr and not tl and not bl and not br:
                        local_id = tile_set[11]  # inner top-right
                    elif bl and not tl and not tr and not br:
                        local_id = tile_set[12]  # inner bottom-left
                    elif br and not tl and not tr and not bl:
                        local_id = tile_set[13]  # inner bottom-right
                    else:
                        # Fully interior cell — skip so decorations can go here.
                        continue
                else:
                    local_id = tile_set[idx + 1]  # +1 because tile_set[0] is flat_below, [1] is center
            else:
                # Fallback: use center tile (has at least one odd-pattern neighbor)
                local_id = tile_set[1]

            if local_id >= 0:
                result[row, col] = ROCK_FIRSTGID + local_id

    return result


# ---------------------------------------------------------------------------
# Step 1: Coastline
# ---------------------------------------------------------------------------

def run_coastline(state: WizardState, preview_cb=None):
    """Subdivide-randomize-mirror loop + scale. Writes randomized_matrix, coastline_height_map."""
    initial_matrix = state.initial_matrix
    height, width = state.height, state.width
    mirroring = state.mirroring

    upscales = [5, 10, 20, 40, 80, 160, 320, 640, 1280]
    num_upscales = 0
    for i in range(len(upscales)):
        if min(height, width) >= upscales[i]:
            num_upscales = i
    num_upscales += 1

    randomized_matrix = np.array(initial_matrix)
    if preview_cb:
        preview_cb("initial_matrix", randomized_matrix.copy())

    for i in range(num_upscales):
        subdivided_matrix = subdivide(randomized_matrix)
        randomized_matrix = mirror(randomize(subdivided_matrix), mirroring)
        if preview_cb:
            preview_cb(f"upscale_{i + 1}/{num_upscales}", randomized_matrix.copy())

    coastline_height_map = scale_matrix(randomized_matrix, height, width)

    state.randomized_matrix = randomized_matrix
    state.coastline_height_map = coastline_height_map.copy()

    # Initialize wall_matrix for step 2
    state.wall_matrix = np.zeros((height, width), dtype=int)

    state.completed_step = max(state.completed_step, int(WizardStep.COASTLINE))

    if preview_cb:
        preview_cb("coastline_complete", coastline_height_map.copy())


# ---------------------------------------------------------------------------
# Step 3: Height/Ocean
# ---------------------------------------------------------------------------

# How far (in tiles) the wall influence extends
WALL_INFLUENCE_RADIUS = 12


def _bias_terrain_near_walls(height_map, wall_matrix, num_height_levels):
    """Boost land terrain levels near walls so terrain visually matches hills.

    Cells adjacent to walls get pushed toward stone/soil, fading to grass further
    away.  Water cells (<=0) are never modified.
    """
    wall_mask = wall_matrix == 1
    # Distance from each cell to the nearest wall cell (chessboard metric)
    dist = distance_transform_cdt(~wall_mask, metric='chessboard').astype(float)

    # Normalize: 1.0 at a wall, 0.0 at WALL_INFLUENCE_RADIUS or beyond
    influence = np.clip(1.0 - dist / WALL_INFLUENCE_RADIUS, 0.0, 1.0)

    # Only affect land cells (height >= 1)
    land_mask = height_map >= 1

    # Target level near walls — cap at the max height level the user configured
    # (e.g. level 5 = stone for default 7 levels)
    max_target = min(num_height_levels, 5)

    # Compute boosted level: blend current level toward max_target based on influence
    current = height_map[land_mask].astype(float)
    boosted = current + influence[land_mask] * (max_target - current)
    # Round to integer levels, but never go below the current level
    boosted = np.clip(np.round(boosted).astype(int), height_map[land_mask], max_target)

    result = height_map.copy()
    result[land_mask] = boosted
    return result


def run_height_ocean(state: WizardState, seed=None, preview_cb=None):
    """Perlin + height/ocean levels. Writes perlin_seed, perlin_map, height_map."""
    height, width = state.height, state.width

    if seed is None:
        seed = random.randint(0, 99999)
    state.perlin_seed = seed

    # Start from the binary coastline
    height_map = state.coastline_height_map.copy()

    perlin_map = perlin(height, width, octaves_num=9, seed=seed)
    state.perlin_map = perlin_map

    num_height_levels = state.num_height_levels
    num_ocean_levels = state.num_ocean_levels

    perlin_change = 1 / num_height_levels
    perlin_value = -0.5

    for level in range(2, num_height_levels + 1):
        perlin_value += perlin_change
        height_map = generate_level(height_map, perlin_map, "height", level=level, min_perlin_value=perlin_value)
        if preview_cb:
            preview_cb(f"height_level_{level}", height_map.copy())

    perlin_change = 1 / num_ocean_levels
    perlin_value = -0.5

    for level in range(-1, -num_ocean_levels - 1, -1):
        perlin_value += perlin_change
        height_map = generate_level(height_map, perlin_map, "ocean", level=level, min_perlin_value=perlin_value)
        if preview_cb:
            preview_cb(f"ocean_level_{level}", height_map.copy())

    # Bias terrain near walls: push land cells toward higher levels close to walls
    if state.wall_matrix is not None and np.any(state.wall_matrix == 1):
        height_map = _bias_terrain_near_walls(height_map, state.wall_matrix, num_height_levels)
        if preview_cb:
            preview_cb("wall_bias", height_map.copy())

    state.height_map = height_map
    state.completed_step = max(state.completed_step, int(WizardStep.HEIGHT_OCEAN))

    if preview_cb:
        preview_cb("height_ocean_complete", height_map.copy())


# ---------------------------------------------------------------------------
# Step 4: Command Centers
# ---------------------------------------------------------------------------

def run_place_cc_random(state: WizardState):
    """Random CC placement avoiding water + walls. Writes units_matrix, cc_positions, cc_groups."""
    randomized_matrix = state.randomized_matrix
    height_map = state.height_map
    mirroring = state.mirroring
    num_centers = state.num_command_centers

    # Build items_matrix from current resource placements
    items_matrix = state.items_matrix if state.items_matrix is not None else np.zeros_like(height_map)

    units_matrix = add_command_centers(
        randomized_matrix, num_centers, mirroring, height_map.shape, items_matrix
    )

    # Extract placed positions
    cc_positions = list(zip(*np.where(units_matrix > 0)))

    state.units_matrix = units_matrix
    state.cc_positions = cc_positions
    state.cc_groups = [cc_positions[:]]  # One group for the whole random placement
    state.completed_step = max(state.completed_step, int(WizardStep.COMMAND_CENTERS))


def run_place_cc_manual(state: WizardState, row, col):
    """Place single CC + mirrors. Returns list of placed (row, col) positions."""
    height_map = state.height_map
    h, w = height_map.shape
    mirroring = state.mirroring
    randomized_matrix = state.randomized_matrix
    rm_h, rm_w = randomized_matrix.shape

    # Map from height_map coords to randomized_matrix coords
    rm_row = int(row * rm_h / h)
    rm_col = int(col * rm_w / w)

    # Validate: must be land and not wall
    if rm_row < 0 or rm_row >= rm_h or rm_col < 0 or rm_col >= rm_w:
        return []
    if randomized_matrix[rm_row, rm_col] != 1:
        return []
    if state.wall_matrix is not None and state.wall_matrix[row, col] == 1:
        return []
    if height_map[row, col] <= 0:
        return []

    # Get mirrored positions in randomized_matrix coords
    mirrored_rm = _get_mirrored_positions(rm_row, rm_col, rm_h, rm_w, mirroring)

    # Scale back to height_map coords
    scale_y = h / rm_h
    scale_x = w / rm_w
    placed = []
    for mr, mc in mirrored_rm:
        sr = int(mr * scale_y)
        sc = int(mc * scale_x)
        sr = min(sr, h - 1)
        sc = min(sc, w - 1)
        placed.append((sr, sc))

    if state.units_matrix is None:
        state.units_matrix = np.zeros((h, w), dtype=int)

    # Assign CC IDs
    existing_max = int(state.units_matrix.max()) if state.units_matrix.max() > 0 else 100
    next_id = max(existing_max + 1, 101)

    for pos in placed:
        if state.units_matrix[pos[0], pos[1]] == 0:
            state.units_matrix[pos[0], pos[1]] = next_id
            next_id += 1

    state.cc_positions.extend(placed)
    state.cc_groups.append(placed)

    return placed


def undo_last_cc(state: WizardState):
    """Remove last CC group, rebuild units_matrix."""
    if not state.cc_groups:
        return

    last_group = state.cc_groups.pop()
    for r, c in last_group:
        if state.units_matrix is not None:
            state.units_matrix[r, c] = 0
        if (r, c) in state.cc_positions:
            state.cc_positions.remove((r, c))


def clear_all_cc(state: WizardState):
    """Remove all CCs."""
    h, w = state.height_map.shape if state.height_map is not None else (state.height, state.width)
    state.units_matrix = np.zeros((h, w), dtype=int)
    state.cc_positions = []
    state.cc_groups = []


# ---------------------------------------------------------------------------
# Step 5: Resources
# ---------------------------------------------------------------------------

def run_place_resources_random(state: WizardState):
    """Random resource placement avoiding water + walls + CCs."""
    randomized_matrix = state.randomized_matrix
    height_map = state.height_map
    mirroring = state.mirroring
    num_resource_pulls = state.num_resource_pulls

    items_matrix = np.zeros_like(height_map)

    height_map_copy, items_matrix = add_resource_pulls(
        randomized_matrix, num_resource_pulls, mirroring, height_map, items_matrix
    )

    resource_positions = list(zip(*np.where(items_matrix > 0)))

    state.items_matrix = items_matrix
    state.resource_positions = resource_positions
    state.resource_groups = [resource_positions[:]]
    state.completed_step = max(state.completed_step, int(WizardStep.RESOURCES))


def run_place_resource_manual(state: WizardState, row, col):
    """Place single resource + mirrors. Returns list of placed (row, col) positions."""
    height_map = state.height_map
    h, w = height_map.shape
    mirroring = state.mirroring
    randomized_matrix = state.randomized_matrix
    rm_h, rm_w = randomized_matrix.shape

    rm_row = int(row * rm_h / h)
    rm_col = int(col * rm_w / w)

    if rm_row < 0 or rm_row >= rm_h or rm_col < 0 or rm_col >= rm_w:
        return []
    if randomized_matrix[rm_row, rm_col] != 1:
        return []
    if state.wall_matrix is not None and state.wall_matrix[row, col] == 1:
        return []
    if height_map[row, col] <= 0:
        return []

    mirrored_rm = _get_mirrored_positions(rm_row, rm_col, rm_h, rm_w, mirroring)
    scale_y = h / rm_h
    scale_x = w / rm_w

    placed = []
    for mr, mc in mirrored_rm:
        sr = int(mr * scale_y)
        sc = int(mc * scale_x)
        sr = min(sr, h - 1)
        sc = min(sc, w - 1)
        placed.append((sr, sc))

    if state.items_matrix is None:
        state.items_matrix = np.zeros((h, w), dtype=int)

    for sr, sc in placed:
        if state.items_matrix[sr, sc] == 0:
            place_resource_pull(state.items_matrix, sr, sc)

    state.resource_positions.extend(placed)
    state.resource_groups.append(placed)

    return placed


def undo_last_resource(state: WizardState):
    """Remove last resource group, rebuild items_matrix."""
    if not state.resource_groups:
        return

    last_group = state.resource_groups.pop()
    for r, c in last_group:
        # Clear 3x3 area around each resource center
        h, w = state.items_matrix.shape
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    state.items_matrix[nr, nc] = 0
        if (r, c) in state.resource_positions:
            state.resource_positions.remove((r, c))


def clear_all_resources(state: WizardState):
    """Remove all resources."""
    h, w = state.height_map.shape if state.height_map is not None else (state.height, state.width)
    state.items_matrix = np.zeros((h, w), dtype=int)
    state.resource_positions = []
    state.resource_groups = []


# ---------------------------------------------------------------------------
# Step 6: Finalize & Export
# ---------------------------------------------------------------------------

def run_finalize(state: WizardState, preview_cb=None):
    """Terrain smoothing + decoration. Writes id_matrix."""
    tile_sets = {
        "water_sand":       (31, 34, 6, 7, 8, 33, 35, 60, 61, 62, 87, 88, 114, 115),
        "sand_grass":       (34, 37, 9, 10, 11, 36, 38, 63, 64, 65, 90, 91, 117, 118),
        "grass_soil":       (37, 40, 12, 13, 14, 39, 41, 66, 67, 68, 93, 94, 120, 121),
        "soil_swamp":       (40, 43, 15, 16, 17, 42, 44, 69, 70, 71, 96, 97, 123, 124),
        "swamp_stone":      (43, 46, 18, 19, 20, 45, 47, 72, 73, 74, 99, 100, 126, 127),
        "stone_snow":       (46, 49, 21, 22, 23, 48, 50, 75, 76, 77, 102, 103, 129, 130),
        "snow_ice":         (49, 52, 24, 25, 26, 51, 53, 78, 79, 80, 105, 106, 132, 133),
        "deep_water_water": (28, 31, 3, 4, 5, 30, 32, 57, 58, 59, 84, 85, 111, 112),
        "ocean_deep_water": (83, 28, 0, 1, 2, 27, 29, 54, 55, 56, 81, 82, 108, 109),
    }

    decoration_tiles = {
        1: (86, 89, 116),
        2: (110, 119),
        3: (95, 122),
        4: (98, 125),
        5: (101, 128),
        6: (104, 131),
        7: (107, 134),
    }

    terrain_levels = [
        (-1, tile_sets["ocean_deep_water"]),
        (0, tile_sets["deep_water_water"]),
        (1, tile_sets["water_sand"]),
        (2, tile_sets["sand_grass"]),
        (3, tile_sets["grass_soil"]),
        (4, tile_sets["soil_swamp"]),
        (5, tile_sets["swamp_stone"]),
        (6, tile_sets["stone_snow"]),
        (7, tile_sets["snow_ice"]),
    ]

    height_map = state.height_map.copy()
    h, w = height_map.shape
    id_matrix = np.full((h, w), 83)

    items_matrix = state.items_matrix if state.items_matrix is not None else np.zeros((h, w), dtype=int)
    units_matrix = state.units_matrix if state.units_matrix is not None else np.zeros((h, w), dtype=int)

    for level, tile_set in reversed(terrain_levels):
        id_matrix = smooth_terrain_tiles(height_map, id_matrix, level, tile_set)
        if preview_cb:
            preview_cb(f"terrain_smooth_{level}", height_map.copy(), id_matrix.copy(),
                       items_matrix.copy(), units_matrix.copy())

    id_matrix = add_decoration_tiles(id_matrix, height_map, decoration_tiles, DECORATION_FREQUENCY)

    # Merge wall tiles into items layer (Rusted Warfare only supports Ground/Items/Units)
    if state.wall_matrix is not None and np.any(state.wall_matrix):
        wall_id_matrix = smooth_wall_tiles(state.wall_matrix)
        if wall_id_matrix is not None:
            # Write wall GIDs into items_matrix where there's no existing item
            wall_mask = wall_id_matrix > 0
            items_matrix[wall_mask] = wall_id_matrix[wall_mask]
            if preview_cb:
                preview_cb("wall_tiles", height_map.copy(), id_matrix.copy(),
                           items_matrix.copy(), units_matrix.copy())

    state.id_matrix = id_matrix
    state.items_matrix = items_matrix
    state.completed_step = max(state.completed_step, int(WizardStep.FINALIZE))

    if preview_cb:
        preview_cb("terrain_complete", height_map.copy(), id_matrix.copy(),
                   items_matrix.copy(), units_matrix.copy())


def write_tmx(state: WizardState):
    """Encode + write TMX file."""
    height_map = state.height_map
    id_matrix = state.id_matrix
    items_matrix = state.items_matrix if state.items_matrix is not None else np.zeros_like(height_map)
    units_matrix = state.units_matrix if state.units_matrix is not None else np.zeros_like(height_map)

    h, w = id_matrix.shape

    tile_data_items = b''.join(int(j).to_bytes(4, 'little') for i in items_matrix for j in i)
    gzip_data_items = gzip.compress(tile_data_items)
    base64_data_items = base64.b64encode(gzip_data_items)

    tile_data_units = b''.join(int(j).to_bytes(4, 'little') for i in units_matrix for j in i)
    gzip_data_units = gzip.compress(tile_data_units)
    base64_data_units = base64.b64encode(gzip_data_units)

    tile_data = b''.join((TILE_ID_OFFSET + int(j)).to_bytes(4, 'little') for i in id_matrix for j in i)
    gzip_data = gzip.compress(tile_data)
    base64_data_ground = base64.b64encode(gzip_data)

    template_path = resource_path(f"generator_blueprint{state.pattern}.tmx")
    tree = ET.parse(template_path)
    root = tree.getroot()

    root.set('width', str(w))
    root.set('height', str(h))

    layer_data_map = {
        'Ground': base64_data_ground.decode('ascii'),
        'Items': base64_data_items.decode('ascii'),
        'Units': base64_data_units.decode('ascii'),
    }

    for layer in root.findall('layer'):
        name = layer.get('name')
        layer.set('width', str(w))
        layer.set('height', str(h))
        if name in layer_data_map:
            data_elem = layer.find('data')
            if data_elem is not None:
                data_elem.text = layer_data_map[name]

    output_file = os.path.join(state.output_path, "generated_map.tmx")
    tree.write(output_file, encoding='UTF-8', xml_declaration=True)
    logger.info(f"Map has been created at: {output_file}")
    return output_file
