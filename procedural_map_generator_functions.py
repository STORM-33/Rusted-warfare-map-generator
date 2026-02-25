import numpy as np
import random
from perlin_noise import PerlinNoise
from collections import deque
from scipy.spatial.distance import cdist

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


def count_neighbors(matrix, i, j):
    neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1), (i-1, j-1), (i-1, j+1), (i+1, j-1), (i+1, j+1)]
    count = 0
    for x, y in neighbors:
        if 0 <= x < len(matrix) and 0 <= y < len(matrix[0]) and matrix[x][y] != matrix[i][j]:
            count += 1
    return count


def subdivide(matrix):
    new_size_i = len(matrix) * 2
    new_size_j = len(matrix[0]) * 2
    new_matrix = [[" " for _ in range(new_size_j)] for _ in range(new_size_i)]

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            current_color = matrix[i][j]
            new_matrix[i * 2][j * 2] = current_color
            new_matrix[i * 2 + 1][j * 2] = current_color
            new_matrix[i * 2][j * 2 + 1] = current_color
            new_matrix[i * 2 + 1][j * 2 + 1] = current_color

    return new_matrix


def randomize(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            neighbors = count_neighbors(matrix, i, j)
            if neighbors >= 3 and random.random() < 0.2 + 0.1 * (neighbors - 3):
                matrix[i][j] = 1 - matrix[i][j]
    return matrix


def generate_level(map_matrix, perlin_noise, level_type, level, min_perlin_value, min_distance_to_prev_level=3, min_distance_to_next_level=4):
    def bfs(map_matrix, start, max_distance, level, level_type):
        rows, cols = map_matrix.shape
        visited = np.zeros((rows, cols), dtype=bool)
        queue = deque([(start, 0)])

        neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        while queue:
            (i, j), dist = queue.popleft()
            if dist > max_distance:
                continue
            if level_type == 'height' and map_matrix[i][j] == level - 2:
                return False
            if level_type == 'ocean' and map_matrix[i][j] == level + 2:
                return False
            for di, dj in neighbor_offsets:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols and not visited[ni][nj]:
                    if abs(ni - start[0]) + abs(nj - start[1]) <= max_distance:
                        visited[ni][nj] = True
                        queue.append(((ni, nj), dist + 1))
        return True

    rows, cols = map_matrix.shape
    new_map = map_matrix.copy()

    for i in range(rows):
        for j in range(cols):
            current_level = map_matrix[i, j]
            if level_type == 'height' and current_level == level - 1:
                min_distance = min_distance_to_prev_level
            elif level_type == 'ocean' and current_level == level + 1:
                min_distance = min_distance_to_next_level
            else:
                continue
            perlin_value = perlin_noise[i, j]
            if perlin_value >= min_perlin_value:
                min_dist_satisfied = bfs(map_matrix, (i, j), min_distance, level, level_type)
                if min_dist_satisfied:
                    new_map[i, j] = level

    return new_map


def mirror(matrix, mirroring):
    if mirroring not in ["none", "horizontal", "vertical", "diagonal1", "diagonal2", "both"]:
        print("Mirroring option was defined incorrectly")
        return matrix

    if mirroring == "none":
        return matrix
    elif mirroring == "horizontal":
        n = len(matrix)
        m = len(matrix[0])
        mid = n // 2
        for i in range(mid):
            for j in range(m):
                matrix[n - 1 - i][j] = matrix[i][j]
        return matrix
    elif mirroring == "vertical":
        n = len(matrix)
        m = len(matrix[0])
        mid = m // 2
        for i in range(n):
            for j in range(mid):
                matrix[i][m - 1 - j] = matrix[i][j]
        return matrix
    elif mirroring == "diagonal1":
        n = len(matrix)
        for i in range(n):
            for j in range(i + 1, n):
                matrix[j][i] = matrix[i][j]
        return matrix
    elif mirroring == "diagonal2":
        return [[matrix[j][i] if j + i < len(matrix) else matrix[len(matrix)-1-i][len(matrix[0])-1-j] for i in range(len(matrix[0]))] for j in range(len(matrix))]
    elif mirroring == "both":
        n = len(matrix)
        m = len(matrix[0])
        mid = n // 2
        for i in range(mid):
            for j in range(m):
                matrix[n - 1 - i][j] = matrix[i][j]
        mid = m // 2
        for i in range(n):
            for j in range(mid):
                matrix[i][m - 1 - j] = matrix[i][j]
        return matrix


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
    src_height = len(matrix)
    src_width = len(matrix[0])
    scaled_matrix = [[" " for _ in range(target_width)] for _ in range(target_height)]

    for i in range(target_height):
        for j in range(target_width):
            src_i = int(i * src_height / target_height)
            src_j = int(j * src_width / target_width)
            scaled_matrix[i][j] = matrix[src_i][src_j]

    return scaled_matrix


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
    placed = []
    for i in range(num_positions):
        if i == 0:
            center_i, center_j = rows // 4, cols // 2
            if (center_i, center_j) in available_tiles:
                placed.append((center_i, center_j))
                available_tiles.remove((center_i, center_j))
            else:
                tile = available_tiles.pop(0)
                placed.append(tile)
        else:
            max_distance = float('-inf')
            best_tile = None
            for tile in available_tiles:
                min_distance = min([abs(tile[0] - x) + abs(tile[1] - y) for x, y in placed], default=float('inf'))
                if min_distance > max_distance:
                    max_distance = min_distance
                    best_tile = tile
            placed.append(best_tile)
            available_tiles.remove(best_tile)
    return placed


def add_resource_pulls(randomized_matrix, num_resource_pulls, mirroring, height_map, items_matrix):
    rows, cols = randomized_matrix.shape

    forbidden_zones = _get_forbidden_zones(rows, cols, mirroring)
    available_tiles = _find_valid_resource_positions(randomized_matrix, forbidden_zones)

    if len(available_tiles) < num_resource_pulls:
        print("Not enough space to place the specified number of resource pulls")
        return items_matrix

    placed_pulls = _select_spaced_positions(available_tiles, num_resource_pulls, rows, cols)

    pull_matrix = np.zeros((rows, cols))
    for i, j in placed_pulls:
        pull_matrix[i][j] = 1

    pull_matrix = mirror(pull_matrix, mirroring)

    scale_factor_x = height_map.shape[1] / randomized_matrix.shape[1]
    scale_factor_y = height_map.shape[0] / randomized_matrix.shape[0]

    for i in range(pull_matrix.shape[0]):
        for j in range(pull_matrix.shape[1]):
            if pull_matrix[i][j] == 1:
                scaled_i = int(i * scale_factor_y)
                scaled_j = int(j * scale_factor_x)
                place_resource_pull(items_matrix, scaled_i, scaled_j)

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
            mirrored_y = height - 1 - y
            scaled_y = int(mirrored_y * scale_y)
            scaled_x = int(x * scale_x)
        elif mirroring == 'vertical':
            mirrored_x = width - 1 - x
            scaled_y = int(y * scale_y)
            scaled_x = int(mirrored_x * scale_x)
        elif mirroring == 'diagonal1':
            scaled_y = int(x * scale_y)
            scaled_x = int(y * scale_x)
        elif mirroring == 'diagonal2':
            mirrored_y = width - 1 - x
            mirrored_x = height - 1 - y
            scaled_y = int(mirrored_y * scale_y)
            scaled_x = int(mirrored_x * scale_x)
        else:
            continue
        scaled_units_matrix[scaled_y, scaled_x] = team1_val + 5

    return scaled_units_matrix


def add_command_centers(randomized_matrix, num_centers, mirroring, height_map_shape):
    if mirroring not in ['horizontal', 'vertical', 'diagonal1', 'diagonal2']:
        return np.zeros(height_map_shape)

    num_centers = num_centers // 2
    height = randomized_matrix.shape[0]
    margin = int(CC_MARGIN_RATIO * height)

    valid_positions, preferred_area = _find_valid_cc_positions(randomized_matrix, mirroring, margin)

    if len(valid_positions) < num_centers:
        raise ValueError("Not enough valid positions for command centers")

    selected_positions = []
    preferred_positions = [pos for pos in valid_positions if preferred_area[pos]]

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


def create_map_matrix(initial_matrix, height, width, mirroring, num_resource_pulls, num_command_centers, num_height_levels, num_ocean_levels, preview_callback=None):
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

    randomized_matrix = initial_matrix
    if preview_callback:
        preview_callback("initial_matrix", np.array(initial_matrix), None, None, None)

    for i in range(num_upscales):
        subdivided_matrix = subdivide(randomized_matrix)
        randomized_matrix = mirror(randomize(subdivided_matrix), mirroring)
        if preview_callback:
            preview_callback(f"upscale_{i + 1}/{num_upscales}", np.array(randomized_matrix), None, None, None)

    randomized_matrix = np.array(randomized_matrix)
    scaled_matrix = scale_matrix(randomized_matrix, height, width)
    print("Basic matrix created")

    perlin_map = perlin(height, width, octaves_num=9, seed=random.randint(0, 99999))
    height_map = np.array(scaled_matrix)

    if preview_callback:
        preview_callback("scaled_matrix", height_map.copy(), None, None, None)

    print("Perlin matrix generated")
    perlin_change = 1/num_height_levels
    perlin_value = -0.5

    for level in range(2, num_height_levels+1):
        perlin_value += perlin_change
        height_map = generate_level(height_map, perlin_map, "height", level=level, min_perlin_value=perlin_value)
        print(f"Level {level} generated")
        if preview_callback:
            preview_callback(f"height_level_{level}", height_map.copy(), None, None, None)

    perlin_change = 1 / num_ocean_levels
    perlin_value = -0.5

    for level in list(range(-1, num_ocean_levels*-1, -1)):
        perlin_value += perlin_change
        height_map = generate_level(height_map, perlin_map, "ocean", level=level, min_perlin_value=perlin_value)
        print(f"Level {level} generated")
        if preview_callback:
            preview_callback(f"ocean_level_{level}", height_map.copy(), None, None, None)

    items_matrix = np.zeros_like(height_map)
    height_map, items_matrix = add_resource_pulls(randomized_matrix, num_resource_pulls, mirroring, height_map, items_matrix)
    print("Resource pulls added")
    if preview_callback:
        preview_callback("resource_pulls", height_map.copy(), None, items_matrix.copy(), None)

    units_matrix = add_command_centers(randomized_matrix, num_command_centers, mirroring, height_map.shape)
    print("Command centers added")
    if preview_callback:
        preview_callback("command_centers", height_map.copy(), None, items_matrix.copy(), units_matrix.copy())

    return height_map, items_matrix, units_matrix
