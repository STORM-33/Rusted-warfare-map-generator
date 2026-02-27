import logging
import numpy as np
import random
from perlin_noise import PerlinNoise
from scipy.spatial.distance import cdist
from scipy.ndimage import distance_transform_cdt

logger = logging.getLogger(__name__)

# --- Constants ---
BORDER_MARGIN_RATIO = 0.08
CENTER_FORBIDDEN_RATIO = 0.06
CC_MARGIN_RATIO = 0.07
CC_MIN_DISTANCE_RATIO = 0.1
DECORATION_FREQUENCY = 0.05

RESOURCE_PULL_TILES = {
    (-1, -1): 1,
    (-1, 0): 2,
    (-1, 1): 3,
    (0, -1): 11,
    (0, 0): 12,
    (0, 1): 13,
    (1, -1): 21,
    (1, 0): 22,
    (1, 1): 23,
}


def subdivide(matrix):
    arr = np.asarray(matrix)
    return np.repeat(np.repeat(arr, 2, axis=0), 2, axis=1)


def randomize(matrix, smoothness=0.0):
    arr = np.asarray(matrix)
    rows, cols = arr.shape
    padded = np.pad(arr, 1, mode='edge')
    # Count differing neighbors (8-connectivity)
    neighbor_count = np.zeros((rows, cols), dtype=int)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            neighbor_count += (padded[1+di:rows+1+di, 1+dj:cols+1+dj] != arr).astype(int)
    # Probabilistic flip where neighbors >= 3, scaled by smoothness
    base_prob = 0.2 + 0.1 * (neighbor_count - 3)
    prob = base_prob * (1.0 - smoothness * 0.85)
    prob = np.clip(prob, 0, 1)
    flip_mask = (neighbor_count >= 3) & (np.random.random((rows, cols)) < prob)
    result = arr.copy()
    result[flip_mask] = 1 - result[flip_mask]
    return result


def generate_level(map_matrix, perlin_noise, level_type, level, min_perlin_value,
                   min_distance_to_prev_level=3, min_distance_to_next_level=4):
    rows, cols = map_matrix.shape
    new_map = map_matrix.copy()

    if level_type == 'height':
        candidate_mask = map_matrix == (level - 1)
        forbidden_mask = map_matrix == (level - 2)
        min_distance = min_distance_to_prev_level
    else:  # ocean
        candidate_mask = map_matrix == (level + 1)
        forbidden_mask = map_matrix == (level + 2)
        min_distance = min_distance_to_next_level

    perlin_mask = perlin_noise >= min_perlin_value

    if forbidden_mask.any():
        dist = distance_transform_cdt(~forbidden_mask, metric='chessboard')
        distance_ok = dist > min_distance
    else:
        distance_ok = np.ones((rows, cols), dtype=bool)

    new_map[candidate_mask & perlin_mask & distance_ok] = level
    return new_map


def mirror(matrix, mirroring):
    arr = np.asarray(matrix)
    if mirroring not in ["none", "horizontal", "vertical", "diagonal1", "diagonal2", "both"]:
        logger.warning("Mirroring option was defined incorrectly")
        return arr

    if mirroring == "none":
        return arr
    elif mirroring == "horizontal":
        mid = arr.shape[0] // 2
        arr[arr.shape[0] - mid:] = arr[:mid][::-1]
        return arr
    elif mirroring == "vertical":
        mid = arr.shape[1] // 2
        arr[:, arr.shape[1] - mid:] = arr[:, :mid][:, ::-1]
        return arr
    elif mirroring == "diagonal1":
        n = arr.shape[0]
        for i in range(n):
            arr[i+1:, i] = arr[i, i+1:n]
        return arr
    elif mirroring == "diagonal2":
        n = arr.shape[0]
        source = arr.copy()
        for i in range(n):
            for j in range(n):
                if j + i >= n:
                    arr[j][i] = source[n - 1 - i][n - 1 - j]
        return arr
    elif mirroring == "both":
        mid_r = arr.shape[0] // 2
        arr[arr.shape[0] - mid_r:] = arr[:mid_r][::-1]
        mid_c = arr.shape[1] // 2
        arr[:, arr.shape[1] - mid_c:] = arr[:, :mid_c][:, ::-1]
        return arr


def get_neighbors(matrix, x, y):
    rows, cols = matrix.shape
    neighbors = [(x-1, y), (x, y+1), (x+1, y), (x, y-1)]
    result = []
    for nx, ny in neighbors:
        if 0 <= nx < rows and 0 <= ny < cols:
            result.append(int(matrix[nx, ny] < matrix[x, y]))
        else:
            result.append(0)
    return tuple(result)


def get_all_neighbors(matrix, x, y):
    rows, cols = matrix.shape
    neighbors = [(x-1, y-1), (x-1, y), (x-1, y+1),
                 (x, y-1), (x, y), (x, y+1),
                 (x+1, y-1), (x+1, y), (x+1, y+1)]
    result = []
    for nx, ny in neighbors:
        if 0 <= nx < rows and 0 <= ny < cols:
            result.append(int(matrix[nx, ny] < matrix[x, y]))
        else:
            result.append(0)
    return tuple(result[i:i+3] for i in range(0, len(result), 3))


# --- Thin/isolated tile patterns that should be removed ---
_ISOLATED_PATTERNS = [
    (1, 0, 1, 0), (1, 1, 1, 0), (0, 1, 0, 1),
    (1, 1, 0, 1), (1, 0, 1, 1), (0, 1, 1, 1), (1, 1, 1, 1)
]

_DIAGONAL_ISOLATED_PATTERNS = (
    ([1, 0, 0], [0, 0, 0], [0, 0, 1]),
    ([0, 0, 1], [0, 0, 0], [1, 0, 0]),
    ([0, 1, 0], [1, 0, 0], [0, 0, 0]),
    ([0, 1, 0], [0, 0, 1], [0, 0, 0]),
    ([0, 0, 0], [1, 0, 0], [0, 1, 0]),
    ([0, 0, 0], [0, 0, 1], [0, 1, 0]),
)

_CARDINAL_TILE_MAP = {
    (0, 0, 0, 0): 0,
    (1, 0, 0, 0): 2,
    (0, 1, 0, 0): 5,
    (1, 1, 0, 0): 3,
    (0, 0, 1, 0): 7,
    (1, 0, 1, 0): -1,
    (0, 1, 1, 0): 8,
    (1, 1, 1, 0): -1,
    (0, 0, 0, 1): 4,
    (1, 0, 0, 1): 1,
    (0, 1, 0, 1): -1,
    (1, 1, 0, 1): -1,
    (0, 0, 1, 1): 6,
    (1, 0, 1, 1): -1,
    (0, 1, 1, 1): -1,
    (1, 1, 1, 1): -1,
}


def _remove_isolated_tiles(map_matrix, height_level, passes):
    height, width = np.shape(map_matrix)
    for _ in range(passes):
        for x in range(height):
            for y in range(width):
                if map_matrix[x][y] == height_level:
                    neighbors = get_neighbors(map_matrix, x, y)
                    if neighbors in _ISOLATED_PATTERNS:
                        map_matrix[x][y] = height_level - 1


def _assign_edge_tiles(map_matrix, id_matrix, height_level, tile_set):
    height, width = np.shape(map_matrix)
    for x in range(height):
        for y in range(width):
            if map_matrix[x][y] == height_level:
                neighbors = get_neighbors(map_matrix, x, y)
                id_matrix[x][y] = int(tile_set[_CARDINAL_TILE_MAP[neighbors] + 1])


def _assign_corner_tiles(map_matrix, id_matrix, height_level, tile_set):
    height, width = np.shape(map_matrix)
    for x in range(height):
        for y in range(width):
            if map_matrix[x][y] == height_level:
                neighbors = get_all_neighbors(map_matrix, x, y)
                if neighbors == ([1, 0, 0], [0, 0, 0], [0, 0, 0]):
                    id_matrix[x][y] = int(tile_set[13])
                elif neighbors == ([0, 0, 1], [0, 0, 0], [0, 0, 0]):
                    id_matrix[x][y] = int(tile_set[12])
                elif neighbors == ([0, 0, 0], [0, 0, 0], [1, 0, 0]):
                    id_matrix[x][y] = int(tile_set[11])
                elif neighbors == ([0, 0, 0], [0, 0, 0], [0, 0, 1]):
                    id_matrix[x][y] = int(tile_set[10])
                elif neighbors == ([0, 0, 1], [0, 0, 1], [1, 0, 0]):
                    id_matrix[x][y] = int(tile_set[9])
                elif neighbors == ([0, 0, 1], [1, 0, 0], [1, 0, 0]):
                    id_matrix[x][y] = int(tile_set[2])
                elif neighbors == ([1, 0, 0], [0, 0, 1], [0, 0, 1]):
                    id_matrix[x][y] = int(tile_set[4])
                elif neighbors == ([1, 0, 0], [1, 0, 0], [0, 0, 1]):
                    id_matrix[x][y] = int(tile_set[7])
                elif neighbors == ([1, 0, 0], [0, 0, 0], [0, 1, 1]):
                    id_matrix[x][y] = int(tile_set[7])
                elif neighbors == ([1, 1, 0], [0, 0, 0], [0, 0, 1]):
                    id_matrix[x][y] = int(tile_set[4])
                elif neighbors == ([0, 0, 1], [0, 0, 0], [1, 1, 0]):
                    id_matrix[x][y] = int(tile_set[9])
                elif neighbors == ([0, 1, 1], [0, 0, 0], [1, 0, 0]):
                    id_matrix[x][y] = int(tile_set[2])


def smooth_terrain_tiles(map_matrix, id_matrix, height_level, tile_set):
    height, width = np.shape(map_matrix)

    # First pass: remove thin/isolated tiles (3 iterations)
    _remove_isolated_tiles(map_matrix, height_level, passes=3)

    # Remove diagonal-only connected tiles
    for x in range(height):
        for y in range(width):
            if map_matrix[x][y] == height_level:
                neighbors = get_all_neighbors(map_matrix, x, y)
                if neighbors in _DIAGONAL_ISOLATED_PATTERNS:
                    map_matrix[x][y] = height_level - 1

    # Second pass: remove remaining isolated tiles (5 iterations)
    _remove_isolated_tiles(map_matrix, height_level, passes=5)

    # Assign edge tiles based on cardinal neighbors
    _assign_edge_tiles(map_matrix, id_matrix, height_level, tile_set)

    # Assign corner tiles based on all neighbors
    _assign_corner_tiles(map_matrix, id_matrix, height_level, tile_set)

    return id_matrix


def perlin(x, y, octaves_num, seed=0):
    noise = PerlinNoise(octaves=octaves_num, seed=seed)
    pic = np.array([[noise([i / x, j / x]) for j in range(y)] for i in range(x)])
    return pic


def scale_matrix(matrix, target_height, target_width):
    arr = np.asarray(matrix)
    src_h, src_w = arr.shape
    row_idx = (np.arange(target_height) * src_h // target_height).astype(int)
    col_idx = (np.arange(target_width) * src_w // target_width).astype(int)
    return arr[np.ix_(row_idx, col_idx)]


def place_resource_pull(items_matrix, i, j):
    for di in range(-1, 2):
        for dj in range(-1, 2):
            ni, nj = i + di, j + dj
            if 0 <= ni < len(items_matrix) and 0 <= nj < len(items_matrix[0]):
                items_matrix[ni][nj] = RESOURCE_PULL_TILES[(di, dj)]


def _get_forbidden_zones(rows, cols, mirroring):
    forbidden_zones = set()
    border_size = int(min(rows, cols) * BORDER_MARGIN_RATIO)
    center_forbidden_size = int(min(rows, cols) * CENTER_FORBIDDEN_RATIO)

    for i in range(rows):
        for j in range(cols):
            if i < border_size or i >= rows - border_size or j < border_size or j >= cols - border_size:
                forbidden_zones.add((i, j))

    if mirroring == "horizontal":
        center = rows // 2
        for i in range(center - center_forbidden_size // 2, center + center_forbidden_size // 2):
            forbidden_zones.update((i, j) for j in range(cols))
    elif mirroring == "vertical":
        center = cols // 2
        for j in range(center - center_forbidden_size // 2, center + center_forbidden_size // 2):
            forbidden_zones.update((i, j) for i in range(rows))
    elif mirroring == "diagonal1":
        for i in range(rows):
            for j in range(max(0, i - center_forbidden_size // 2), min(cols, i + center_forbidden_size // 2)):
                forbidden_zones.add((i, j))
    elif mirroring == "diagonal2":
        for i in range(rows):
            for j in range(max(0, cols - 1 - i - center_forbidden_size // 2),
                           min(cols, cols - 1 - i + center_forbidden_size // 2)):
                forbidden_zones.add((i, j))
    elif mirroring == "both":
        center_row, center_col = rows // 2, cols // 2
        for i in range(center_row - center_forbidden_size // 2, center_row + center_forbidden_size // 2):
            forbidden_zones.update((i, j) for j in range(cols))
        for j in range(center_col - center_forbidden_size // 2, center_col + center_forbidden_size // 2):
            forbidden_zones.update((i, j) for i in range(rows))

    return forbidden_zones


def _find_valid_resource_positions(randomized_matrix, forbidden_zones):
    rows, cols = randomized_matrix.shape
    available_tiles = []
    for i in range(2, rows - 2):
        for j in range(2, cols - 2):
            if randomized_matrix[i][j] == 1 and (i, j) not in forbidden_zones:
                valid = True
                for di in range(-2, 3):
                    for dj in range(-2, 3):
                        ni, nj = i + di, j + dj
                        if randomized_matrix[ni][nj] != randomized_matrix[i][j]:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    available_tiles.append((i, j))
    return available_tiles


def _select_spaced_positions(available_tiles, num_positions, rows, cols):
    tiles = np.array(available_tiles)
    placed_idx = []

    # First: pick a random tile
    first = random.randint(0, len(tiles) - 1)
    placed_idx.append(first)

    for _ in range(1, num_positions):
        placed = tiles[placed_idx]
        dists = np.abs(tiles[:, None] - placed[None, :]).sum(axis=2).min(axis=1)
        dists[placed_idx] = -1

        # Pick randomly from the top candidates weighted by distance
        valid_mask = dists > 0
        valid_dists = dists[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        weights = valid_dists / valid_dists.sum()
        chosen = np.random.choice(valid_indices, p=weights)
        placed_idx.append(chosen)

    return [tuple(tiles[i]) for i in placed_idx]


def _get_mirrored_positions(i, j, rows, cols, mirroring):
    """Return all positions (including original) that a point maps to after mirroring."""
    positions = [(i, j)]
    if mirroring == "horizontal":
        positions.append((rows - 1 - i, j))
    elif mirroring == "vertical":
        positions.append((i, cols - 1 - j))
    elif mirroring == "diagonal1":
        positions.append((j, i))
    elif mirroring == "diagonal2":
        positions.append((rows - 1 - j, cols - 1 - i))
    elif mirroring == "both":
        positions.append((rows - 1 - i, j))
        positions.append((i, cols - 1 - j))
        positions.append((rows - 1 - i, cols - 1 - j))
    # Deduplicate (positions on mirror axis map to themselves)
    return list(set(positions))


def _is_valid_pool_position(scaled_i, scaled_j, height_map, placed_positions, min_pool_distance=4):
    """Check if a 3x3 pool can be placed at the given position on the height map."""
    h_rows, h_cols = height_map.shape
    if scaled_i - 1 < 0 or scaled_i + 1 >= h_rows or scaled_j - 1 < 0 or scaled_j + 1 >= h_cols:
        return False
    pool_area = height_map[scaled_i - 1:scaled_i + 2, scaled_j - 1:scaled_j + 2]
    if np.any(pool_area <= 0):
        return False
    for pi, pj in placed_positions:
        if abs(scaled_i - pi) < min_pool_distance and abs(scaled_j - pj) < min_pool_distance:
            return False
    return True


def _find_mirror_axis_positions(randomized_matrix, mirroring):
    """Find valid positions that lie exactly on the mirror axis (map to themselves)."""
    rows, cols = randomized_matrix.shape
    axis_tiles = []

    if mirroring == "horizontal":
        i = rows // 2
        for j in range(2, cols - 2):
            axis_tiles.append((i, j))
    elif mirroring == "vertical":
        j = cols // 2
        for i in range(2, rows - 2):
            axis_tiles.append((i, j))
    elif mirroring == "diagonal1":
        for i in range(2, min(rows, cols) - 2):
            axis_tiles.append((i, i))
    elif mirroring == "diagonal2":
        for i in range(2, min(rows, cols) - 2):
            axis_tiles.append((i, cols - 1 - i))
    elif mirroring == "both":
        # Center point only
        axis_tiles.append((rows // 2, cols // 2))

    # Filter to land tiles with valid 5x5 surroundings
    valid = []
    for i, j in axis_tiles:
        if i < 2 or i >= rows - 2 or j < 2 or j >= cols - 2:
            continue
        if randomized_matrix[i][j] != 1:
            continue
        all_land = True
        for di in range(-2, 3):
            for dj in range(-2, 3):
                if randomized_matrix[i + di][j + dj] != 1:
                    all_land = False
                    break
            if not all_land:
                break
        if all_land:
            valid.append((i, j))
    return valid


def add_resource_pulls(randomized_matrix, num_resource_pulls, mirroring, height_map, items_matrix):
    rows, cols = randomized_matrix.shape

    forbidden_zones = _get_forbidden_zones(rows, cols, mirroring)
    available_tiles = _find_valid_resource_positions(randomized_matrix, forbidden_zones)

    if not available_tiles:
        logger.warning("No valid positions for resource pulls")
        return height_map, items_matrix

    scale_factor_x = height_map.shape[1] / cols
    scale_factor_y = height_map.shape[0] / rows

    # Shuffle candidates for randomness
    random.shuffle(available_tiles)

    placed_positions = []  # scaled positions of all placed pools (including mirrored)

    for ci, cj in available_tiles:
        if len(placed_positions) >= num_resource_pulls:
            break

        remaining = num_resource_pulls - len(placed_positions)

        # Compute all mirrored positions for this candidate
        mirrored = _get_mirrored_positions(ci, cj, rows, cols, mirroring)
        scaled_positions = [(int(mi * scale_factor_y), int(mj * scale_factor_x)) for mi, mj in mirrored]

        # Skip if this group would overshoot the target
        if len(scaled_positions) > remaining:
            continue

        # All mirrored copies must be valid
        all_valid = all(
            _is_valid_pool_position(si, sj, height_map, placed_positions)
            for si, sj in scaled_positions
        )
        if not all_valid:
            continue

        # Place all mirrored copies
        for si, sj in scaled_positions:
            placed_positions.append((si, sj))
            place_resource_pull(items_matrix, si, sj)

    # Second pass: fill remaining slots with on-axis single pools
    if len(placed_positions) < num_resource_pulls and mirroring != "none":
        axis_tiles = _find_mirror_axis_positions(randomized_matrix, mirroring)
        random.shuffle(axis_tiles)
        for ci, cj in axis_tiles:
            if len(placed_positions) >= num_resource_pulls:
                break
            si = int(ci * scale_factor_y)
            sj = int(cj * scale_factor_x)
            if _is_valid_pool_position(si, sj, height_map, placed_positions):
                placed_positions.append((si, sj))
                place_resource_pull(items_matrix, si, sj)

    if len(placed_positions) < num_resource_pulls:
        logger.warning(f"Could only place {len(placed_positions)} of {num_resource_pulls} resource pulls")

    return height_map, items_matrix


def _find_valid_cc_positions(randomized_matrix, mirroring, margin):
    height, width = randomized_matrix.shape

    if mirroring == 'horizontal':
        valid_area = np.zeros_like(randomized_matrix, dtype=bool)
        valid_area[margin:height // 2 - margin, margin:-margin] = True
        preferred_area = valid_area.copy()
        preferred_area[margin:margin * 2, :] = True
    elif mirroring == 'vertical':
        valid_area = np.zeros_like(randomized_matrix, dtype=bool)
        valid_area[margin:-margin, margin:width // 2 - margin] = True
        preferred_area = valid_area.copy()
        preferred_area[:, margin:margin * 2] = True
    elif mirroring == 'diagonal1':
        valid_area = np.triu(np.ones_like(randomized_matrix, dtype=bool), k=margin)
        valid_area[-margin:, :] = False
        valid_area[:, -margin:] = False
        preferred_area = np.zeros_like(randomized_matrix, dtype=bool)
        preferred_area[margin:height // 4, -width // 4:-margin] = True
    elif mirroring == 'diagonal2':
        valid_area = np.fliplr(np.triu(np.ones_like(randomized_matrix, dtype=bool), k=margin))
        valid_area[-margin:, :] = False
        valid_area[:, :margin] = False
        preferred_area = np.zeros_like(randomized_matrix, dtype=bool)
        preferred_area[margin:height // 4, margin:width // 4] = True
    elif mirroring == 'both':
        valid_area = np.zeros_like(randomized_matrix, dtype=bool)
        valid_area[margin:height // 2 - margin, margin:width // 2 - margin] = True
        preferred_area = valid_area.copy()
        preferred_area[margin:margin * 2, margin:margin * 2] = True
    elif mirroring == 'none':
        valid_area = np.zeros_like(randomized_matrix, dtype=bool)
        valid_area[margin:-margin, margin:-margin] = True
        preferred_area = valid_area.copy()
    else:
        return [], None

    valid_positions = np.where((randomized_matrix == 1) & valid_area)

    def is_valid_position(pos):
        y, x = pos
        region = randomized_matrix[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
        return (region == 1).all()

    valid_positions = [(y, x) for y, x in zip(*valid_positions) if is_valid_position((y, x))]
    return valid_positions, preferred_area


def _mirror_command_centers(selected_positions, mirroring, randomized_matrix, height_map_shape):
    height, width = randomized_matrix.shape
    scale_y = height_map_shape[0] / height
    scale_x = height_map_shape[1] / width

    scaled_units_matrix = np.zeros(height_map_shape, dtype=int)

    # Place command centers for team 1
    for i, (y, x) in enumerate(selected_positions):
        scaled_y = int(y * scale_y)
        scaled_x = int(x * scale_x)
        scaled_units_matrix[scaled_y, scaled_x] = 101 + i

    # Mirror for team 2
    for y, x in selected_positions:
        team1_val = scaled_units_matrix[int(y * scale_y), int(x * scale_x)]
        if mirroring == 'horizontal':
            mirrors = [(height - 1 - y, x)]
        elif mirroring == 'vertical':
            mirrors = [(y, width - 1 - x)]
        elif mirroring == 'diagonal1':
            mirrors = [(x, y)]
        elif mirroring == 'diagonal2':
            mirrors = [(width - 1 - x, height - 1 - y)]
        elif mirroring == 'both':
            mirrors = [
                (height - 1 - y, x),
                (y, width - 1 - x),
                (height - 1 - y, width - 1 - x),
            ]
        elif mirroring == 'none':
            continue
        else:
            continue
        for my, mx in mirrors:
            sy = int(my * scale_y)
            sx = int(mx * scale_x)
            scaled_units_matrix[sy, sx] = team1_val + 5

    return scaled_units_matrix


def add_command_centers(randomized_matrix, num_centers, mirroring, height_map_shape, items_matrix=None):
    if mirroring not in ['horizontal', 'vertical', 'diagonal1', 'diagonal2', 'both', 'none']:
        logger.warning(f"Unsupported mirroring mode for command centers: {mirroring}")
        return np.zeros(height_map_shape, dtype=int)

    if mirroring == 'both':
        num_centers = num_centers // 4
    elif mirroring != 'none':
        num_centers = num_centers // 2
    height = randomized_matrix.shape[0]
    margin = int(CC_MARGIN_RATIO * height)

    valid_positions, preferred_area = _find_valid_cc_positions(randomized_matrix, mirroring, margin)

    # Filter out positions that overlap with resource pools
    if items_matrix is not None:
        scale_y = height_map_shape[0] / randomized_matrix.shape[0]
        scale_x = height_map_shape[1] / randomized_matrix.shape[1]
        cc_clearance = 2  # tiles of clearance around a CC to avoid resource pools

        def _overlaps_resource_pool(pos):
            sy = int(pos[0] * scale_y)
            sx = int(pos[1] * scale_x)
            y_min = max(0, sy - cc_clearance)
            y_max = min(items_matrix.shape[0], sy + cc_clearance + 1)
            x_min = max(0, sx - cc_clearance)
            x_max = min(items_matrix.shape[1], sx + cc_clearance + 1)
            return np.any(items_matrix[y_min:y_max, x_min:x_max] != 0)

        valid_positions = [p for p in valid_positions if not _overlaps_resource_pool(p)]
        preferred_positions = [pos for pos in valid_positions if preferred_area[pos]]
    else:
        preferred_positions = [pos for pos in valid_positions if preferred_area[pos]]

    if len(valid_positions) < num_centers:
        raise ValueError("Not enough valid positions for command centers")

    selected_positions = []

    while len(selected_positions) < num_centers:
        if preferred_positions and np.random.random() < 0.7:
            pos = preferred_positions.pop(np.random.randint(len(preferred_positions)))
        else:
            pos = valid_positions.pop(np.random.randint(len(valid_positions)))

        selected_positions.append(pos)

        valid_positions = [p for p in valid_positions if cdist([pos], [p])[0][0] > height * CC_MIN_DISTANCE_RATIO]
        preferred_positions = [p for p in preferred_positions if cdist([pos], [p])[0][0] > height * CC_MIN_DISTANCE_RATIO]

    return _mirror_command_centers(selected_positions, mirroring, randomized_matrix, height_map_shape)


def add_decoration_tiles(id_matrix, map_matrix, dec_tiles, freq):
    height, width = np.shape(map_matrix)
    for i in range(height):
        for j in range(width):
            rand = random.random()
            if get_all_neighbors(map_matrix, i, j) == ([0, 0, 0], [0, 0, 0], [0, 0, 0]) and freq > rand and map_matrix[i][j] > 0:
                id_matrix[i][j] = random.choice(dec_tiles[map_matrix[i][j]])
    return id_matrix


def create_map_matrix(initial_matrix, height, width, mirroring, num_resource_pulls, num_command_centers, num_height_levels, num_ocean_levels, shoreline_smoothness=0.0, preview_callback=None):
    if not isinstance(initial_matrix, list):
        raise ValueError("Initial matrix was defined incorrectly")
    if mirroring not in ["none", "horizontal", "vertical", "diagonal1", "diagonal2", "both"]:
        raise ValueError("Mirroring option was defined incorrectly")
    if num_command_centers % 2 != 0 or num_command_centers > 10:
        raise ValueError("Number of command centers must be an even number up to 10")
    if num_height_levels < 1 or num_height_levels > 7:
        raise ValueError("Number of height levels must be from 1 to 7")
    if num_ocean_levels < 1 or num_ocean_levels > 3:
        raise ValueError("Number of ocean levels must be from 1 to 3")

    upscales = [5, 10, 20, 40, 80, 160, 320, 640, 1280]
    for i in range(len(upscales)):
        if min(height, width) >= upscales[i]:
            num_upscales = i
    num_upscales += 1

    randomized_matrix = np.array(initial_matrix)
    if preview_callback:
        preview_callback("initial_matrix", randomized_matrix.copy(), None, None, None)

    for i in range(num_upscales):
        subdivided_matrix = subdivide(randomized_matrix)
        randomized_matrix = mirror(randomize(subdivided_matrix, smoothness=shoreline_smoothness), mirroring)
        if preview_callback:
            preview_callback(f"upscale_{i + 1}/{num_upscales}", randomized_matrix.copy(), None, None, None)

    height_map = scale_matrix(randomized_matrix, height, width)
    logger.info("Basic matrix created")

    perlin_map = perlin(height, width, octaves_num=9, seed=random.randint(0, 99999))

    if preview_callback:
        preview_callback("scaled_matrix", height_map.copy(), None, None, None)

    logger.info("Perlin matrix generated")
    perlin_change = 1/num_height_levels
    perlin_value = -0.5

    for level in range(2, num_height_levels+1):
        perlin_value += perlin_change
        height_map = generate_level(height_map, perlin_map, "height", level=level, min_perlin_value=perlin_value)
        logger.info(f"Level {level} generated")
        if preview_callback:
            preview_callback(f"height_level_{level}", height_map.copy(), None, None, None)

    perlin_change = 1 / num_ocean_levels
    perlin_value = -0.5

    for level in list(range(-1, -num_ocean_levels - 1, -1)):
        perlin_value += perlin_change
        height_map = generate_level(height_map, perlin_map, "ocean", level=level, min_perlin_value=perlin_value)
        logger.info(f"Level {level} generated")
        if preview_callback:
            preview_callback(f"ocean_level_{level}", height_map.copy(), None, None, None)

    items_matrix = np.zeros_like(height_map)
    height_map, items_matrix = add_resource_pulls(randomized_matrix, num_resource_pulls, mirroring, height_map, items_matrix)
    logger.info("Resource pulls added")
    if preview_callback:
        preview_callback("resource_pulls", height_map.copy(), None, items_matrix.copy(), None)

    units_matrix = add_command_centers(randomized_matrix, num_command_centers, mirroring, height_map.shape, items_matrix)
    logger.info("Command centers added")
    if preview_callback:
        preview_callback("command_centers", height_map.copy(), None, items_matrix.copy(), units_matrix.copy())

    return height_map, items_matrix, units_matrix
