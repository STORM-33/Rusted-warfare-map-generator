import gzip
import base64
from procedural_map_generator_functions import *
import numpy as np
import os


def generate_map(initial_matrix,
                 height, width,
                 mirroring,
                 num_res_pulls,
                 num_com_centers,
                 num_height_levels,
                 num_ocean_levels,
                 pattern,
                 output_path):

    tile_sets = {"water_sand":       (31, 34, 6, 7, 8, 33, 35, 60, 61, 62, 87, 88, 114, 115),
                 "sand_grass":       (34, 37, 9, 10, 11, 36, 38, 63, 64, 65, 90, 91, 117, 118),
                 "grass_soil":       (37, 40, 12, 13, 14, 39, 41, 66, 67, 68, 93, 94, 120, 121),
                 "soil_swamp":       (40, 43, 15, 16, 17, 42, 44, 69, 70, 71, 96, 97, 123, 124),
                 "swamp_stone":      (43, 46, 18, 19, 20, 45, 47, 72, 73, 74, 99, 100, 126, 127),
                 "stone_snow":       (46, 49, 21, 22, 23, 48, 50, 75, 76, 77, 102, 103, 129, 130),
                 "snow_ice":         (49, 52, 24, 25, 26, 51, 53, 78, 79, 80, 105, 106, 132, 133),
                 "deep_water_water": (28, 31, 3, 4, 5, 30, 32, 57, 58, 59, 84, 85, 111, 112),
                 "ocean_deep_water": (83, 28, 0, 1, 2, 27, 29, 54, 55, 56, 81, 82, 108, 109)}

    decoration_tiles = {1: (86, 89, 116),
                        2: (110, 119),
                        3: (95, 122),
                        4: (98, 125),
                        5: (101, 128),
                        6: (104, 131),
                        7: (107, 134)}

    ID = range(0, 10000)
    map_matrix, items_matrix, units_matrix = create_map_matrix(initial_matrix=initial_matrix,
                                                               height=height, width=width,
                                                               mirroring=mirroring,
                                                               num_res_pulls=num_res_pulls,
                                                               num_com_centers=num_com_centers,
                                                               num_height_levels=num_height_levels,
                                                               num_ocean_levels=num_ocean_levels)
    height, width = np.shape(map_matrix)

    tile_data = []
    tile_data_items = []
    tile_data_units = []

    for i in items_matrix:
        for j in i:
            tile_data_items.append(ID[int(j)].to_bytes(4, 'little'))

    gzip_data_items = gzip.compress(b''.join(tile_data_items))
    base64_data_items = base64.b64encode(gzip_data_items)

    for i in units_matrix:
        for j in i:
            tile_data_units.append(ID[int(j)].to_bytes(4, 'little'))

    gzip_data_units = gzip.compress(b''.join(tile_data_units))
    base64_data_units = base64.b64encode(gzip_data_units)

    terrain_levels = [
        (-1, tile_sets["ocean_deep_water"]),
        (-0, tile_sets["deep_water_water"]),
        (1, tile_sets["water_sand"]),
        (2, tile_sets["sand_grass"]),
        (3, tile_sets["grass_soil"]),
        (4, tile_sets["soil_swamp"]),
        (5, tile_sets["swamp_stone"]),
        (6, tile_sets["stone_snow"]),
        (7, tile_sets["snow_ice"]),
    ]

    id_matrix = np.full((height, width), 83)

    for level, tile_set in reversed(terrain_levels):
        id_matrix = fcking_smoothing_function1(map_matrix, id_matrix, level, tile_set)
    id_matrix = add_decoration_tiles(id_matrix, map_matrix, decoration_tiles, 0.05)

    for i in id_matrix:
        for j in i:
            tile_data.append(ID[201+int(j)].to_bytes(4, 'little'))

    gzip_data = gzip.compress(b''.join(tile_data))
    base64_data_ground = base64.b64encode(gzip_data)

    with open(f"generator_blueprint{pattern}.tmx", "r", encoding='utf-8') as map_file:
        file = map_file.readlines()

        output_file = os.path.join(output_path, "generated_map.tmx")
        with open(output_file, "w", encoding='utf-8') as new_map:
            for line in file:
                if "<map version=\"1.2\" orientation=\"orthogonal\" renderorder=" in line:
                    line = f"<map version=\"1.2\" orientation=\"orthogonal\" renderorder=\"right-down\" width=\"{width}\" height=\"{height}\" tilewidth=\"20\" tileheight=\"20\" nextobjectid=\"1\">"
                elif "<layer name=\"Ground\"" in line:
                    line = f"  <layer name=\"Ground\" width=\"{width}\" height=\"{height}\">"
                elif "<layer name=\"Items\"" in line:
                    line = f"  <layer name=\"Items\" width=\"{width}\" height=\"{height}\">"
                elif "<layer name=\"Units\"" in line:
                    line = f"  <layer name=\"Units\" width=\"{width}\" height=\"{height}\">"
                elif "Ground layer data" in line:
                    line = str(base64_data_ground)[2:-1]
                elif "Items layer data" in line:
                    line = str(base64_data_items)[2:-1]
                elif "Units layer data" in line:
                    line = str(base64_data_units)[2:-1]
                new_map.write(line)
            print(f"Map has been created at: {output_file}")


def main():
    initial_matrix = [[1, 0, 1, 0, 1], [1, 1, 0, 1, 1], [1, 0, 1, 0, 1], [1, 1, 0, 1, 1], [1, 0, 1, 0, 1]]
    generate_map(initial_matrix=initial_matrix,
                 height=160, width=160,
                 mirroring="diagonal1",
                 num_res_pulls=24,
                 num_com_centers=10,
                 num_height_levels=7,
                 num_ocean_levels=3,
                 pattern=5,
                 output_path="")


if __name__ == "__main__":
    main()

# gzip_data = gzip.compress(b''.join(tile_data))
# base64_data_ground = base64.b64encode(gzip_data)
