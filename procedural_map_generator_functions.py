import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
from perlin_noise import PerlinNoise
from collections import deque
from scipy.spatial.distance import cdist


# Функция для подсчета соседей другого цвета
def count_neighbors(matrix, i, j):
    neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1), (i-1, j-1), (i-1, j+1), (i+1, j-1), (i+1, j+1)]
    count = 0
    for x, y in neighbors:
        if 0 <= x < len(matrix) and 0 <= y < len(matrix[0]) and matrix[x][y] != matrix[i][j]:
            count += 1
    return count



# Функция для умножения площади матрицы в четыре раза
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

# Функция для изменения тайлов на основе количества окружающих тайлов противоположного цвета
def randomize(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            neighbors = count_neighbors(matrix, i, j)
            if neighbors >= 3 and random.random() < 0.2 + 0.1 * (neighbors - 3):
                matrix[i][j] = 1 - matrix[i][j]
    return matrix




#функция для генерации нового уровня высоты поверх уровня на 1 ниже
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

# Функция для отражения матрицы
def mirror(matrix, mirroring_option):
    if mirroring_option in ["none", "horizontal", "vertical", "diagonal1", "diagonal2", "4-corners"]:
        if mirroring_option == "none":
            return matrix
        elif mirroring_option == "horizontal":
            n = len(matrix)
            m = len(matrix[0])
            mid = n // 2
            for i in range(mid):
                for j in range(m):
                    matrix[n - 1 - i][j] = matrix[i][j]
            return matrix
        elif mirroring_option == "vertical":
            n = len(matrix)
            m = len(matrix[0])
            mid = m // 2
            for i in range(n):
                for j in range(mid):
                    matrix[i][m - 1 - j] = matrix[i][j]
            return matrix
        elif mirroring_option == "diagonal1":
            n = len(matrix)
            for i in range(n):
                for j in range(i + 1, n):
                    matrix[j][i] = matrix[i][j]
            return matrix
        elif mirroring_option == "diagonal2":
            return [[matrix[j][i] if j + i < len(matrix) else matrix[len(matrix)-1-i][len(matrix[0])-1-j] for i in range(len(matrix[0]))] for j in range(len(matrix))]
        elif mirroring_option == "4-corners":
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
    else:
        print("Mirroring option was defined incorrectly")
        return matrix

def visualize(matrix):
    # Создаем цветовую карту: 0 - зеленый, 1 - синий
    cmap = ListedColormap(['blue', 'green'])

    # Создаем фигуру и оси
    fig, ax = plt.subplots()

    # Отображаем матрицу с заданной цветовой картой
    ax.imshow(matrix, cmap=cmap)

    # Убираем оси
    ax.axis('off')

    # Показываем график
    plt.show()


def get_neighbors(matrix, x, y):
    # Получаем размер матрицы
    rows, cols = matrix.shape

    # Определяем соседей для данной ячейки
    neighbors = [(x-1, y), (x, y+1), (x+1, y), (x, y-1)]

    # Инициализируем результат
    result = []

    # Проверяем каждого соседа
    for nx, ny in neighbors:
        # Если сосед находится внутри границ матрицы
        if 0 <= nx < rows and 0 <= ny < cols:
            # Если значение соседа больше, добавляем 1, иначе добавляем 0
            result.append(int(matrix[nx, ny] < matrix[x, y]))
        else:
            # Если соседа нет (он находится за границей матрицы), добавляем 0
            result.append(0)

    return tuple(result)


def get_all_neighbors(matrix, x, y):
    # Получаем размер матрицы
    rows, cols = matrix.shape

    # Определяем всех соседей для данной ячейки
    neighbors = [(x-1, y-1), (x-1, y), (x-1, y+1),
                 (x, y-1), (x, y), (x, y+1),
                 (x+1, y-1), (x+1, y), (x+1, y+1)]

    # Инициализируем результат
    result = []

    # Проверяем каждого соседа
    for nx, ny in neighbors:
        # Если сосед находится внутри границ матрицы
        if 0 <= nx < rows and 0 <= ny < cols:
            # Добавляем 1, если значение соседа больше, и 0 в противном случае
            result.append(int(matrix[nx, ny] < matrix[x, y]))
        else:
            # Если соседа нет (он находится за границей матрицы), добавляем 0
            result.append(0)

    # Преобразуем список в кортеж кортежей размером 3x3
    return tuple(result[i:i+3] for i in range(0, len(result), 3))

def fcking_smoothing_function1(map_matrix, id_matrix, height_level, tile_set):

    neighbors_variants={(0, 0, 0, 0): 0,
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
                       (1, 1, 1, 1): -1}

    height, width = np.shape(map_matrix)
    for i in range(3):
        for x in range(height):
            for y in range(width):
                if map_matrix[x][y] == height_level:
                    neighbors = get_neighbors(map_matrix, x, y)
                    if neighbors in [(1, 0, 1, 0), (1, 1, 1, 0), (0, 1, 0, 1), (1, 1, 0, 1), (1, 0, 1, 1), (0, 1, 1, 1), (1, 1, 1, 1)]:
                        map_matrix[x][y] = height_level - 1

    for x in range(height):
        for y in range(width):
            if map_matrix[x][y] == height_level:
                neighbors = get_all_neighbors(map_matrix, x, y)
                if neighbors in (([1, 0, 0], [0, 0, 0], [0, 0, 1]),
                                 ([0, 0, 1], [0, 0, 0], [1, 0, 0]),
                                 ([0, 1, 0], [1, 0, 0], [0, 0, 0]),
                                 ([0, 1, 0], [0, 0, 1], [0, 0, 0]),
                                 ([0, 0, 0], [1, 0, 0], [0, 1, 0]),
                                 ([0, 0, 0], [0, 0, 1], [0, 1, 0])):
                    map_matrix[x][y] = height_level - 1

    for i in range(5):
        for x in range(height):
            for y in range(width):
                if map_matrix[x][y] == height_level:
                    neighbors = get_neighbors(map_matrix, x, y)
                    if neighbors in [(1, 0, 1, 0), (1, 1, 1, 0), (0, 1, 0, 1), (1, 1, 0, 1), (1, 0, 1, 1), (0, 1, 1, 1), (1, 1, 1, 1)]:
                        map_matrix[x][y] = height_level - 1

    for x in range(height):
        for y in range(width):
            if map_matrix[x][y] == height_level:
                neighbors = get_neighbors(map_matrix, x, y)
                id_matrix[x][y] = int(tile_set[neighbors_variants[neighbors]+1])

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

    return id_matrix


def visualize_height_map(map_matrix):
    """
    Визуализирует матрицу карты с использованием matplotlib.pyplot.

    Args:
        map_matrix (numpy.ndarray): Матрица карты с уровнями высот и глубин.

    Returns:
        None
    """
    # Определение цветовой палитры
    colors = [(0, 0, 0.6),      # темно-синий (-2)
              (0, 0, 0.8),      # синий (-1)
              (0, 0, 1),        # синий (0)
              (1, 1, 0),        # желтый (1)
              (0.5, 0.8, 0.5),  # светло-зеленый (2)
              (0, 0.5, 0),      # зеленый (3)
              (0.8, 0.6, 0),    # темно-желтый (4)
              (0.8, 0.6, 0.4),  # светло-коричневый (5)
              (0.5, 0.2, 0),    # коричневый (6)
              (0.5, 0.5, 0.5)]  # серый (7)


    cmap = ListedColormap(colors)

    # Создание изображения из матрицы
    fig, ax = plt.subplots(figsize=(10, 10))
    img = ax.imshow(map_matrix, cmap=cmap)

    # Добавление цветовой легенды
    cbar = fig.colorbar(img, ticks=range(-2, 8))
    cbar.ax.set_yticklabels(['deep ocean', 'ocean', 'water', 'sand', 'height 2', 'height 3', 'height 4', 'height 5', 'height 6', 'height 7'])

    # Отображение изображения
    plt.show()

def perlin(x, y, octaves_num, seed=0):
    # значения от -0.5 до 0.5
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
    items_matrix[i][j] = 12
    for di in range(-1, 2):
        for dj in range(-1, 2):
            ni, nj = i + di, j + dj
            if 0 <= ni < len(items_matrix) and 0 <= nj < len(items_matrix[0]):
                if di == -1 and dj == -1:
                    items_matrix[ni][nj] = 1
                elif di == -1 and dj == 0:
                    items_matrix[ni][nj] = 2
                elif di == -1 and dj == 1:
                    items_matrix[ni][nj] = 3
                elif di == 0 and dj == -1:
                    items_matrix[ni][nj] = 11
                elif di == 0 and dj == 1:
                    items_matrix[ni][nj] = 13
                elif di == 1 and dj == -1:
                    items_matrix[ni][nj] = 21
                elif di == 1 and dj == 0:
                    items_matrix[ni][nj] = 22
                elif di == 1 and dj == 1:
                    items_matrix[ni][nj] = 23


def add_resource_pulls(randomized_matrix, num_resource_pulls, mirroring, height_map, items_matrix):
    rows, cols = randomized_matrix.shape
    available_tiles = []
    pull_coordinates = []

    # Определение запрещенных зон в зависимости от типа отражения
    forbidden_zones = set()
    border_size = int(min(rows, cols) * 0.08)
    center_forbidden_size = int(min(rows, cols) * 0.06)  # 6% от длины карты

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
    elif mirroring == "4-corners":
        center_row, center_col = rows // 2, cols // 2
        for i in range(center_row - center_forbidden_size // 2, center_row + center_forbidden_size // 2):
            forbidden_zones.update((i, j) for j in range(cols))
        for j in range(center_col - center_forbidden_size // 2, center_col + center_forbidden_size // 2):
            forbidden_zones.update((i, j) for i in range(rows))

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

    if len(available_tiles) < num_resource_pulls:
        print("Недостаточно места для размещения заданного количества экстракторов")
        return items_matrix

    placed_pulls = []
    pull_matrix = np.zeros((rows, cols))

    for i in range(num_resource_pulls):
        if i == 0:
            center_i, center_j = rows // 4, cols // 2
            if (center_i, center_j) in available_tiles:
                pull_matrix[center_i][center_j] = 1
                placed_pulls.append((center_i, center_j))
                available_tiles.remove((center_i, center_j))
            else:
                tile = available_tiles.pop(0)
                pull_matrix[tile[0]][tile[1]] = 1
                placed_pulls.append(tile)
        else:
            max_distance = float('-inf')
            best_tile = None
            for tile in available_tiles:
                min_distance = min([abs(tile[0] - x) + abs(tile[1] - y) for x, y in placed_pulls], default=float('inf'))
                if min_distance > max_distance:
                    max_distance = min_distance
                    best_tile = tile
            pull_matrix[best_tile[0]][best_tile[1]] = 1
            placed_pulls.append(best_tile)
            available_tiles.remove(best_tile)

    pull_matrix = mirror(pull_matrix, mirroring)

    # Масштабирование координат экстракторов
    scale_factor_x = height_map.shape[1] / randomized_matrix.shape[1]
    scale_factor_y = height_map.shape[0] / randomized_matrix.shape[0]

    for i in range(pull_matrix.shape[0]):
        for j in range(pull_matrix.shape[1]):
            if pull_matrix[i][j] == 1:
                scaled_i = int(i * scale_factor_y)
                scaled_j = int(j * scale_factor_x)
                place_resource_pull(items_matrix, scaled_i, scaled_j)

    return height_map, items_matrix


def add_com_centers(randomized_matrix, num_centers, mirror_type, height_map_shape):
    if mirror_type not in ['none', 'horizontal', 'vertical', 'diagonal1', 'diagonal2']:
        return np.zeros(height_map_shape)

    if mirror_type == 'none':
        return np.zeros(height_map_shape)

    num_centers = num_centers // 2
    height, width = randomized_matrix.shape
    units_matrix = np.zeros_like(randomized_matrix)

    # Calculate border margin (7% of map length)
    margin = int(0.07 * height)

    # Define valid area and preferred area for team 1
    if mirror_type == 'horizontal':
        valid_area = np.zeros_like(randomized_matrix, dtype=bool)
        valid_area[margin:height // 2 - margin, margin:-margin] = True
        preferred_area = valid_area.copy()
        preferred_area[margin:margin * 2, :] = True
    elif mirror_type == 'vertical':
        valid_area = np.zeros_like(randomized_matrix, dtype=bool)
        valid_area[margin:-margin, margin:width // 2 - margin] = True
        preferred_area = valid_area.copy()
        preferred_area[:, margin:margin * 2] = True
    elif mirror_type == 'diagonal1':
        valid_area = np.triu(np.ones_like(randomized_matrix, dtype=bool), k=margin)
        valid_area[-margin:, :] = False
        valid_area[:, -margin:] = False
        preferred_area = np.zeros_like(randomized_matrix, dtype=bool)
        preferred_area[margin:height // 4, -width // 4:-margin] = True
    elif mirror_type == 'diagonal2':
        valid_area = np.fliplr(np.triu(np.ones_like(randomized_matrix, dtype=bool), k=margin))
        valid_area[-margin:, :] = False
        valid_area[:, :margin] = False
        preferred_area = np.zeros_like(randomized_matrix, dtype=bool)
        preferred_area[margin:height // 4, margin:width // 4] = True

    # Find valid positions
    valid_positions = np.where(
        (randomized_matrix == 1) &
        valid_area
    )

    # Function to check if a position is valid for a command center
    def is_valid_position(pos):
        y, x = pos
        region = randomized_matrix[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
        return (region == 1).all()

    # Filter valid positions
    valid_positions = [(y, x) for y, x in zip(*valid_positions) if is_valid_position((y, x))]

    if len(valid_positions) < num_centers:
        raise ValueError("Not enough valid positions for command centers")

    # Select positions with preference to the preferred area
    selected_positions = []
    preferred_positions = [pos for pos in valid_positions if preferred_area[pos]]

    while len(selected_positions) < num_centers:
        if preferred_positions and np.random.random() < 0.7:  # 70% chance to pick from preferred area
            pos = preferred_positions.pop(np.random.randint(len(preferred_positions)))
        else:
            pos = valid_positions.pop(np.random.randint(len(valid_positions)))

        selected_positions.append(pos)

        # Remove nearby positions to maintain some distance
        valid_positions = [p for p in valid_positions if cdist([pos], [p])[0][0] > height * 0.1]
        preferred_positions = [p for p in preferred_positions if cdist([pos], [p])[0][0] > height * 0.1]

    # Scale factor for coordinates
    scale_y = height_map_shape[0] / randomized_matrix.shape[0]
    scale_x = height_map_shape[1] / randomized_matrix.shape[1]

    # Create scaled units matrix
    scaled_units_matrix = np.zeros(height_map_shape, dtype=int)

    # Place and scale command centers for team 1
    for i, (y, x) in enumerate(selected_positions):
        scaled_y = int(y * scale_y)
        scaled_x = int(x * scale_x)
        scaled_units_matrix[scaled_y, scaled_x] = 101 + i

    # Mirror and scale command centers for team 2
    if mirror_type == 'horizontal':
        for y, x in selected_positions:
            mirrored_y = height - 1 - y
            scaled_y = int(mirrored_y * scale_y)
            scaled_x = int(x * scale_x)
            scaled_units_matrix[scaled_y, scaled_x] = scaled_units_matrix[int(y * scale_y), int(x * scale_x)] + 5
    elif mirror_type == 'vertical':
        for y, x in selected_positions:
            mirrored_x = width - 1 - x
            scaled_y = int(y * scale_y)
            scaled_x = int(mirrored_x * scale_x)
            scaled_units_matrix[scaled_y, scaled_x] = scaled_units_matrix[int(y * scale_y), int(x * scale_x)] + 5
    elif mirror_type == 'diagonal1':
        for y, x in selected_positions:
            scaled_y = int(x * scale_y)
            scaled_x = int(y * scale_x)
            scaled_units_matrix[scaled_y, scaled_x] = scaled_units_matrix[int(y * scale_y), int(x * scale_x)] + 5
    elif mirror_type == 'diagonal2':
        for y, x in selected_positions:
            mirrored_y = width - 1 - x
            mirrored_x = height - 1 - y
            scaled_y = int(mirrored_y * scale_y)
            scaled_x = int(mirrored_x * scale_x)
            scaled_units_matrix[scaled_y, scaled_x] = scaled_units_matrix[int(y * scale_y), int(x * scale_x)] + 5

    return scaled_units_matrix

def add_decoration_tiles(id_matrix, map_matrix, dec_tiles, freq):
    height, width = np.shape(map_matrix)
    for i in range(height):
        for j in range(width):
            rand = random.random()
            if get_all_neighbors(map_matrix, i, j) == ([0, 0, 0], [0, 0, 0], [0, 0, 0]) and freq > rand and map_matrix[i][j] > 0:
                id_matrix[i][j] = random.choice(dec_tiles[map_matrix[i][j]])
    return id_matrix


def create_map_matrix(initial_matrix, height, width, mirroring, num_res_pulls, num_com_centers, num_height_levels, num_ocean_levels):
    if not isinstance(initial_matrix, list):
        raise ValueError("Initial matrix was defined incorrectly")
    if mirroring not in ["none", "horizontal", "vertical", "diagonal1", "diagonal2", "4-corners"]:
        raise ValueError("Mirroring option was defined incorrectly")
    if num_com_centers % 2 != 0 or num_com_centers > 10:
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
    for i in range(num_upscales):
        subdivided_matrix = subdivide(randomized_matrix)
        randomized_matrix = mirror(randomize(subdivided_matrix), mirroring)

    randomized_matrix = np.array(randomized_matrix)
    scaled_matrix = scale_matrix(randomized_matrix, height, width)
    print("Basic matrix created")

    perlin_map = perlin(height, width, octaves_num=9, seed=int(random.random() * 1000))
    height_map = np.array(scaled_matrix)

    print("Perlin matrix generated")
    perlin_change = 1/num_height_levels
    perlin_value = -0.5

    for level in range(2, num_height_levels+1):
        perlin_value += perlin_change
        height_map = generate_level(height_map, perlin_map, "height", level=level, min_perlin_value=perlin_value)
        print(f"Level {level} generated")

    perlin_change = 1 / num_ocean_levels
    perlin_value = -0.5

    for level in list(range(-1, num_ocean_levels*-1, -1)):
        perlin_value += perlin_change
        height_map = generate_level(height_map, perlin_map, "ocean", level=level, min_perlin_value=perlin_value)
        print(f"Level {level} generated")

    items_matrix = np.zeros_like(height_map)
    height_map, items_matrix = add_resource_pulls(randomized_matrix, num_res_pulls, mirroring, height_map, items_matrix)
    print("Resource pulls added")

    units_matrix = add_com_centers(randomized_matrix, num_com_centers, mirroring, height_map.shape)
    print("CC added")



    return height_map, items_matrix, units_matrix


def main():
    perlin_map = perlin(200, 200, octaves_num=20, seed=int(random.random() * 1000))
    visualize(perlin_map)


if __name__ == "__main__":
    main()

