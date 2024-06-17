import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
from perlin_noise import PerlinNoise
from collections import deque


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

def fcking_smothing_function1(map_matrix, id_matrix, height_level, tile_set):

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


def add_resource_pulls(height_map, num_resource_pulls):
    rows, cols = height_map.shape
    available_tiles = []
    items_matrix = np.zeros((height_map.shape))

    # Находим все подходящие тайлы в верхней половине карты
    for i in range(2, rows // 2 - 2):
        for j in range(2, cols - 2):
            if height_map[i][j] >= 1:
                valid = True
                for di in range(-2, 3):
                    for dj in range(-2, 3):
                        ni, nj = i + di, j + dj
                        if height_map[ni][nj] != height_map[i][j]:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    available_tiles.append((i, j))

    # Если нет подходящих тайлов, выводим сообщение и возвращаем исходную карту
    if len(available_tiles) < num_resource_pulls:
        print("Недостаточно места для размещения заданного количества экстракторов")
        return height_map

    # Размещаем ресурсные точки в верхней половине карты
    placed_pulls = []
    for i in range(num_resource_pulls):
        # Размещаем первую ресурсную точку в центре верхней половины карты
        if i == 0:
            center_i, center_j = rows // 4, cols // 2
            if (center_i, center_j) in available_tiles:
                place_resource_pull(items_matrix, center_i, center_j)
                placed_pulls.append((center_i, center_j))
                available_tiles.remove((center_i, center_j))
            else:
                tile = available_tiles.pop(0)
                place_resource_pull(items_matrix, tile[0], tile[1])
                placed_pulls.append(tile)
        # Размещаем остальные ресурсные точки равномерно
        else:
            max_distance = float('-inf')
            best_tile = None
            for tile in available_tiles:
                min_distance = min([abs(tile[0] - x) + abs(tile[1] - y) for x, y in placed_pulls], default=float('inf'))
                if min_distance > max_distance:
                    max_distance = min_distance
                    best_tile = tile
            place_resource_pull(items_matrix, best_tile[0], best_tile[1])
            placed_pulls.append(best_tile)
            available_tiles.remove(best_tile)

    # Отзеркаливаем ресурсные точки на нижнюю половину карты
    for i, j in placed_pulls:
        mirror_i = rows - i - 1
        mirror_j = j

        # Проверяем цвета вокруг отзеркаленной точки
        colors = set()
        for di in range(-1, 2):
            for dj in range(-1, 2):
                ni, nj = mirror_i + di, mirror_j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    colors.add(height_map[ni][nj])

        # Если цвета не одинаковые, вычисляем преобладающий цвет
        if len(colors) > 1:
            predominant_color = max(colors, key=lambda x: sum(1 for di in range(-1, 2) for dj in range(-1, 2) if
                                                              0 <= mirror_i + di < rows and 0 <= mirror_j + dj < cols and
                                                              height_map[mirror_i + di][mirror_j + dj] == x))

            # Изменяем отличающиеся цвета на преобладающий
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    ni, nj = mirror_i + di, mirror_j + dj
                    if 0 <= ni < rows and 0 <= nj < cols and height_map[ni][nj] != predominant_color:
                        height_map[ni][nj] = predominant_color

        place_resource_pull(items_matrix, mirror_i, mirror_j)

    return height_map, items_matrix


def add_command_centers(height_map, items_matrix, num_of_command_centers_in_one_team):
    rows, cols = height_map.shape
    units_matrix = np.zeros((rows, cols), dtype=int)
    team1_ids = [2285, 2286, 2287, 2288, 2289]
    team2_ids = [2290, 2291, 2292, 2293, 2294]

    # Находим все подходящие тайлы в верхней четверти карты
    available_tiles = []
    border_distance = int(rows * 0.1)  # Изменено с 0.2 на 0.1
    for i in range(border_distance, rows // 4):  # Изменено с rows // 2 на rows // 4
        for j in range(border_distance, cols - border_distance):
            if height_map[i][j] >= 1 and items_matrix[i][j] == 0:
                valid = True
                for di in range(-2, 3):
                    for dj in range(-2, 3):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows // 4 and 0 <= nj < cols and (  # Изменено с rows // 2 на rows // 4
                                height_map[ni][nj] != height_map[i][j] or items_matrix[ni][nj] != 0):
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    available_tiles.append((i, j))

    # Вычисляем минимальное расстояние между командными центрами
    min_distance = int(cols / num_of_command_centers_in_one_team)

    # Сортируем available_tiles по возрастанию расстояния до верхней границы
    available_tiles.sort(key=lambda x: x[0])

    # Размещаем командные центры команды 1 в верхней половине карты
    placed_centers_team1 = []
    for i in range(num_of_command_centers_in_one_team):
        best_tile = None

        # Ищем тайл, удовлетворяющий минимальному расстоянию до уже размещенных центров
        for tile in available_tiles:
            distance_to_centers = min([abs(tile[0] - x) + abs(tile[1] - y) for x, y in placed_centers_team1],
                                      default=float('inf'))
            if distance_to_centers >= min_distance:
                best_tile = tile
                break

        # Если не найден подходящий тайл, выбираем ближайший к верхней границе
        if best_tile is None:
            if available_tiles:
                best_tile = available_tiles[0]
            else:
                print("Недостаточно места для размещения командного центра")
                break

        units_matrix[best_tile[0]][best_tile[1]] = team1_ids[i]
        placed_centers_team1.append(best_tile)
        available_tiles.remove(best_tile)


    #отзеркаливаем координаты коммандных центров
    for i, (j, k) in enumerate(placed_centers_team1):
        mirror_i = rows - j - 1
        mirror_j = k

        # Изменяем значения вокруг отзеркаленной точки на преобладающее значение в области 3х3
        values = []
        for di in range(-1, 2):
            for dj in range(-1, 2):
                ni, nj = mirror_i + di, mirror_j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    values.append(height_map[ni][nj])

        predominant_value = max(set(values), key=values.count)
        for di in range(-1, 2):
            for dj in range(-1, 2):
                ni, nj = mirror_i + di, mirror_j + dj
                if 0 <= ni < rows and 0 <= nj < cols and items_matrix[ni][nj] == 0:
                    height_map[ni][nj] = predominant_value

        if i < len(team2_ids) and items_matrix[mirror_i][mirror_j] == 0:
            units_matrix[mirror_i][mirror_j] = team2_ids[i]

    return height_map, units_matrix



def create_map_matrix(initial_matrix, height, width, mirroring):
    randomized_matrix = initial_matrix
    for i in range(6):
        subdivided_matrix = subdivide(randomized_matrix)
        randomized_matrix = mirror(randomize(subdivided_matrix), mirroring)
        m = np.array(randomized_matrix)

    scaled_matrix = scale_matrix(randomized_matrix, height, width)
    perlin_map = perlin(height, width, octaves_num=5, seed=int(random.random() * 1000))

    height_map2 = generate_level(np.array(scaled_matrix), perlin_map, "height", level=2, min_perlin_value=-0.3)
    height_map3 = generate_level(height_map2, perlin_map, "height", level=3, min_perlin_value=-0.1)
    height_map4 = generate_level(height_map3, perlin_map, "height", level=4, min_perlin_value=0.1)
    height_map5 = generate_level(height_map4, perlin_map, "height", level=5, min_perlin_value=-0.25)
    height_map6 = generate_level(height_map5, perlin_map, "height", level=6, min_perlin_value=-0.35)
    height_map7 = generate_level(height_map6, perlin_map, "height", level=7, min_perlin_value=-0.45)
    height_map8 = generate_level(height_map7, perlin_map, "ocean", level=-1, min_perlin_value=-0.3)
    height_map9 = generate_level(height_map8, perlin_map, "ocean", level=-2, min_perlin_value=0.2)
    height_map10, items_matrix = add_resource_pulls(height_map9, 9)
    height_map11, units_matrix = add_command_centers(height_map10, items_matrix, 3)

    return height_map11, items_matrix, units_matrix


def main():
    initial_matrix = [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]]
    height_map, items_matrix, units_matrix = create_map_matrix(initial_matrix, 200, 200, "4-corners")
    visualize_height_map(height_map)


if __name__ == "__main__":
    main()

