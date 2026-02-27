import gzip
import base64
import numpy as np
import os
import sys
import xml.etree.ElementTree as ET


def resource_path(relative_path):
    """Resolve path to bundled data file (PyInstaller-compatible)."""
    base = getattr(sys, '_MEIPASS', os.path.abspath('.'))
    return os.path.join(base, relative_path)
from dataclasses import dataclass
from typing import Optional

from procedural_map_generator_functions import (
    create_map_matrix,
    smooth_terrain_tiles,
    add_decoration_tiles,
    DECORATION_FREQUENCY,
)

TILE_ID_OFFSET = 201


@dataclass
class PreviewState:
    stage: str
    height_map: np.ndarray
    id_matrix: Optional[np.ndarray] = None
    items_matrix: Optional[np.ndarray] = None
    units_matrix: Optional[np.ndarray] = None


def generate_map(initial_matrix,
                 height, width,
                 mirroring,
                 num_resource_pulls,
                 num_command_centers,
                 num_height_levels,
                 num_ocean_levels,
                 pattern,
                 output_path,
                 shoreline_smoothness=0.0,
                 preview_callback=None):

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

    map_matrix, items_matrix, units_matrix = create_map_matrix(initial_matrix=initial_matrix,
                                                               height=height, width=width,
                                                               mirroring=mirroring,
                                                               num_resource_pulls=num_resource_pulls,
                                                               num_command_centers=num_command_centers,
                                                               num_height_levels=num_height_levels,
                                                               num_ocean_levels=num_ocean_levels,
                                                               shoreline_smoothness=shoreline_smoothness,
                                                               preview_callback=preview_callback)
    height, width = np.shape(map_matrix)

    tile_data_items = b''.join(int(j).to_bytes(4, 'little') for i in items_matrix for j in i)
    gzip_data_items = gzip.compress(tile_data_items)
    base64_data_items = base64.b64encode(gzip_data_items)

    tile_data_units = b''.join(int(j).to_bytes(4, 'little') for i in units_matrix for j in i)
    gzip_data_units = gzip.compress(tile_data_units)
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
        id_matrix = smooth_terrain_tiles(map_matrix, id_matrix, level, tile_set)
        if preview_callback:
            preview_callback(f"terrain_smooth_{level}", map_matrix.copy(), id_matrix.copy(), items_matrix.copy(), units_matrix.copy())

    id_matrix = add_decoration_tiles(id_matrix, map_matrix, decoration_tiles, DECORATION_FREQUENCY)

    if preview_callback:
        preview_callback("terrain_complete", map_matrix.copy(), id_matrix.copy(), items_matrix.copy(), units_matrix.copy())

    tile_data = b''.join((TILE_ID_OFFSET + int(j)).to_bytes(4, 'little') for i in id_matrix for j in i)
    gzip_data = gzip.compress(tile_data)
    base64_data_ground = base64.b64encode(gzip_data)

    template_path = resource_path(f"generator_blueprint{pattern}.tmx")
    tree = ET.parse(template_path)
    root = tree.getroot()

    root.set('width', str(width))
    root.set('height', str(height))

    layer_data_map = {
        'Ground': base64_data_ground.decode('ascii'),
        'Items': base64_data_items.decode('ascii'),
        'Units': base64_data_units.decode('ascii'),
    }

    for layer in root.findall('layer'):
        name = layer.get('name')
        layer.set('width', str(width))
        layer.set('height', str(height))
        if name in layer_data_map:
            data_elem = layer.find('data')
            if data_elem is not None:
                data_elem.text = layer_data_map[name]

    output_file = os.path.join(output_path, "generated_map.tmx")
    tree.write(output_file, encoding='UTF-8', xml_declaration=True)
    print(f"Map has been created at: {output_file}")


def main():
    initial_matrix = [[1, 0, 1, 0, 1], [1, 1, 0, 1, 1], [1, 0, 1, 0, 1], [1, 1, 0, 1, 1], [1, 0, 1, 0, 1]]
    generate_map(initial_matrix=initial_matrix,
                 height=160, width=160,
                 mirroring="diagonal1",
                 num_resource_pulls=24,
                 num_command_centers=10,
                 num_height_levels=7,
                 num_ocean_levels=3,
                 pattern=5,
                 output_path="")


if __name__ == "__main__":
    main()
