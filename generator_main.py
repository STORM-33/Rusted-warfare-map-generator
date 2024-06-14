import gzip
import base64
from procedural_map_generator_functions import *
import numpy as np

tile_sets = {"water_sand":       (2, 3, 39, 40, 41, 78, 80, 117, 118, 119, 156, 157, 195, 196),
             "sand_grass":       (3, 4, 432, 433, 434, 471, 473, 510, 511, 512, 549, 550, 588, 589),
             "grass_soil":       (4, 5, 633, 634, 635, 672, 674, 711, 712, 713, 750, 751, 789, 790),
             "soil_swamp":       (5, 6, 1023, 1024, 1025, 1062, 1064, 1101, 1102, 1103, 1140, 1141, 1179, 1180),
             "swamp_stone":      (6, 7, 1338, 1300, 1339, 1262, 1260, 1377, 1222, 1378, 1221, 1223, 1299, 1301),
             "stone_ice":        (7, 8, 627, 628, 629, 666, 668, 705, 706, 707, 744, 745, 783, 784),
             "ice_snow":         (8, 9, 1134, 1096, 1135, 1058, 1056, 1173, 1018, 1174, 1017, 1019, 1095, 1097),
             "deep_water_water": (1, 2, 1329, 1291, 1330, 1253, 1251, 1368, 1213, 1369, 1212, 1214, 1290, 1292),
             "ocean_deep_water": (0, 1, 1326, 1288, 1327, 1250, 1248, 1365, 1210, 1366, 1209, 1211, 1287, 1289)}


initial_matrix = [[0, 1, 1, 0], [1, 1, 1, 1]]

ID = range(0, 10000)
map_matrix, items_matrix, units_matrix = create_map_matrix(initial_matrix, 128, 180, "horizontal")
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
    (6, tile_sets["stone_ice"]),
    (7, tile_sets["ice_snow"]),
]

id_matrix = np.zeros((height, width))

for level, tile_set in reversed(terrain_levels):
    id_matrix = fcking_smothing_function1(map_matrix, id_matrix, level, tile_set)


for i in id_matrix:
    for j in i:
        tile_data.append(ID[3805+int(j)].to_bytes(4, 'little'))

gzip_data = gzip.compress(b''.join(tile_data))
base64_data_ground = base64.b64encode(gzip_data)

with open("generator_blueprint.tmx", "r") as map_file:
    file = map_file.readlines()
    with open(f"C:\\Program Files (x86)\\Steam\\steamapps\\common\\Rusted Warfare\\mods\\maps\\generated_map.tmx", "w") as new_map:
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
        print("Map has been created!")


# gzip_data = gzip.compress(b''.join(tile_data))
# base64_data_ground = base64.b64encode(gzip_data)
