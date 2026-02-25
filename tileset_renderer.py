import xml.etree.ElementTree as ET
import base64
import io
import numpy as np
from PIL import Image
from PyQt5.QtGui import QImage


# Height level -> flat ground tile local ID (tile used when all neighbors are same level)
LEVEL_FLAT_TILES = {
    -2: 83,   # ocean base (initial fill)
    -1: 28,   # deep water
     0: 31,   # water
     1: 34,   # sand
     2: 37,   # grass
     3: 40,   # soil
     4: 43,   # swamp
     5: 46,   # stone
     6: 49,   # snow
     7: 52,   # ice
}


class TilesetExtractor:
    """Parses a TMX blueprint and extracts tile images from embedded PNGs."""

    def __init__(self, tmx_path):
        self._ground_tiles = {}   # local_id -> 20x20 RGBA numpy array
        self._items_tiles = {}
        self._units_tiles = {}
        self._parse_tmx(tmx_path)
        self._ground_center_pixels = None
        self._items_center_pixels = None
        self._units_center_pixels = None

    def _parse_tmx(self, tmx_path):
        tree = ET.parse(tmx_path)
        root = tree.getroot()

        for tileset_elem in root.findall('tileset'):
            name = tileset_elem.get('name')
            columns = int(tileset_elem.get('columns'))
            tilecount = int(tileset_elem.get('tilecount'))
            tilewidth = int(tileset_elem.get('tilewidth', 20))
            tileheight = int(tileset_elem.get('tileheight', 20))

            # Find embedded_png property
            png_b64 = None
            props = tileset_elem.find('properties')
            if props is not None:
                for prop in props.findall('property'):
                    if prop.get('name') == 'embedded_png':
                        png_b64 = prop.text
                        break

            if png_b64 is None:
                continue

            # Decode base64 PNG to PIL Image
            png_data = base64.b64decode(png_b64.strip())
            img = Image.open(io.BytesIO(png_data)).convert('RGBA')
            img_array = np.array(img)

            rows = (tilecount + columns - 1) // columns
            tiles = {}
            for tid in range(tilecount):
                r = tid // columns
                c = tid % columns
                y0 = r * tileheight
                x0 = c * tilewidth
                tile_img = img_array[y0:y0 + tileheight, x0:x0 + tilewidth].copy()
                tiles[tid] = tile_img

            if name == 'AutoLight':
                self._ground_tiles = tiles
            elif name == 'export_items':
                self._items_tiles = tiles
            elif name == '50pCommandCenter':
                self._units_tiles = tiles

    def get_ground_tile(self, local_id):
        return self._ground_tiles.get(local_id)

    def get_items_tile(self, local_id):
        return self._items_tiles.get(local_id)

    def get_units_tile(self, local_id):
        return self._units_tiles.get(local_id)

    @property
    def ground_center_pixels(self):
        if self._ground_center_pixels is None:
            self._ground_center_pixels = self._build_center_lut(self._ground_tiles)
        return self._ground_center_pixels

    @property
    def items_center_pixels(self):
        if self._items_center_pixels is None:
            self._items_center_pixels = self._build_center_lut(self._items_tiles)
        return self._items_center_pixels

    @property
    def units_center_pixels(self):
        if self._units_center_pixels is None:
            self._units_center_pixels = self._build_center_lut(self._units_tiles)
        return self._units_center_pixels

    @staticmethod
    def _build_center_lut(tiles_dict):
        if not tiles_dict:
            return np.zeros((0, 4), dtype=np.uint8)
        max_id = max(tiles_dict.keys())
        lut = np.zeros((max_id + 1, 4), dtype=np.uint8)
        for tid, tile_arr in tiles_dict.items():
            cy, cx = tile_arr.shape[0] // 2, tile_arr.shape[1] // 2
            lut[tid] = tile_arr[cy, cx]
        return lut


class MapRenderer:
    """Renders map matrices to QImage using tileset data or solid colors."""

    MAX_DISPLAY_PX = 512

    def __init__(self, extractor: TilesetExtractor):
        self.extractor = extractor
        self._level_colors = None

    @property
    def level_colors(self):
        """Compute median RGBA color per height level from actual tileset tiles."""
        if self._level_colors is None:
            self._level_colors = {}
            for level, tile_id in LEVEL_FLAT_TILES.items():
                tile = self.extractor.get_ground_tile(tile_id)
                if tile is not None:
                    median = np.median(tile.reshape(-1, 4), axis=0).astype(np.uint8)
                    self._level_colors[level] = tuple(median)
                else:
                    self._level_colors[level] = (128, 128, 128, 255)
        return self._level_colors

    def render_height_map(self, height_map):
        """Render height_map as solid-color 1px-per-tile, scaled to display size."""
        h, w = height_map.shape
        img = np.zeros((h, w, 4), dtype=np.uint8)
        for level, color in self.level_colors.items():
            mask = height_map == level
            img[mask] = color
        return self._numpy_to_qimage_scaled(img)

    def render_terrain(self, id_matrix, items_matrix=None, units_matrix=None):
        """Render tileset-accurate terrain preview."""
        h, w = id_matrix.shape
        if max(h, w) <= 384:
            img = self._render_full(id_matrix, items_matrix, units_matrix)
        else:
            img = self._render_sampled(id_matrix, items_matrix, units_matrix)
        return self._numpy_to_qimage_scaled(img)

    def _render_full(self, id_matrix, items_matrix, units_matrix):
        """Full 20px tile compositing for small maps."""
        h, w = id_matrix.shape

        # Build ground tile atlas for vectorized lookup
        ground_tiles = self.extractor._ground_tiles
        if ground_tiles:
            max_id = max(ground_tiles.keys())
            atlas = np.zeros((max_id + 1, 20, 20, 4), dtype=np.uint8)
            for tid, tile in ground_tiles.items():
                atlas[tid] = tile

            ids = id_matrix.astype(np.intp).clip(0, max_id)
            # ids shape: (h, w) -> tiles shape: (h, w, 20, 20, 4)
            tiles = atlas[ids.ravel()].reshape(h, w, 20, 20, 4)
            # Rearrange to (h*20, w*20, 4)
            img = tiles.transpose(0, 2, 1, 3, 4).reshape(h * 20, w * 20, 4).copy()
        else:
            img = np.zeros((h * 20, w * 20, 4), dtype=np.uint8)

        # Items layer (sparse — keep loop, skip zeros)
        if items_matrix is not None:
            for y in range(h):
                for x in range(w):
                    gid = int(items_matrix[y, x])
                    if gid > 0:
                        local_id = gid - 1
                        tile = self.extractor.get_items_tile(local_id)
                        if tile is not None:
                            self._alpha_composite_tile(img, tile, y * 20, x * 20)

        # Units layer (sparse — keep loop, skip zeros)
        if units_matrix is not None:
            for y in range(h):
                for x in range(w):
                    gid = int(units_matrix[y, x])
                    if gid > 0:
                        local_id = gid - 101
                        tile = self.extractor.get_units_tile(local_id)
                        if tile is not None:
                            self._alpha_composite_tile(img, tile, y * 20, x * 20)

        return img

    @staticmethod
    def _alpha_composite_tile(dst, src, y0, x0):
        """Alpha-composite src tile onto dst at (y0, x0)."""
        th, tw = src.shape[:2]
        region = dst[y0:y0+th, x0:x0+tw]
        alpha_s = src[:, :, 3:4].astype(np.float32) / 255.0
        alpha_d = region[:, :, 3:4].astype(np.float32) / 255.0
        out_a = alpha_s + alpha_d * (1.0 - alpha_s)
        mask = out_a > 0
        out_rgb = np.where(mask,
                           (src[:, :, :3].astype(np.float32) * alpha_s +
                            region[:, :, :3].astype(np.float32) * alpha_d * (1.0 - alpha_s)) / np.where(mask, out_a, 1),
                           0)
        region[:, :, :3] = out_rgb.astype(np.uint8)
        region[:, :, 3] = (out_a * 255).astype(np.uint8)[:, :, 0]

    def _render_sampled(self, id_matrix, items_matrix, units_matrix):
        """Sampled rendering using center pixel LUTs — fast for large maps."""
        h, w = id_matrix.shape
        ground_lut = self.extractor.ground_center_pixels
        items_lut = self.extractor.items_center_pixels
        units_lut = self.extractor.units_center_pixels

        # Ground: id_matrix values are 0-based local IDs
        ids = id_matrix.astype(np.intp)
        ids = np.clip(ids, 0, len(ground_lut) - 1)
        img = ground_lut[ids]  # (h, w, 4)

        # Items layer
        if items_matrix is not None and len(items_lut) > 0:
            item_ids = items_matrix.astype(np.intp)
            mask = item_ids > 0
            if mask.any():
                local_ids = np.clip(item_ids - 1, 0, len(items_lut) - 1)
                item_pixels = items_lut[local_ids]
                alpha = item_pixels[:, :, 3:4].astype(np.float32) / 255.0
                blended = (item_pixels[:, :, :3].astype(np.float32) * alpha +
                           img[:, :, :3].astype(np.float32) * (1.0 - alpha))
                img_copy = img.copy()
                img_copy[mask, :3] = blended[mask].astype(np.uint8)
                img_copy[mask, 3] = 255
                img = img_copy

        # Units layer
        if units_matrix is not None and len(units_lut) > 0:
            unit_ids = units_matrix.astype(np.intp)
            mask = unit_ids > 0
            if mask.any():
                local_ids = np.clip(unit_ids - 101, 0, len(units_lut) - 1)
                unit_pixels = units_lut[local_ids]
                alpha = unit_pixels[:, :, 3:4].astype(np.float32) / 255.0
                blended = (unit_pixels[:, :, :3].astype(np.float32) * alpha +
                           img[:, :, :3].astype(np.float32) * (1.0 - alpha))
                img_copy = img.copy()
                img_copy[mask, :3] = blended[mask].astype(np.uint8)
                img_copy[mask, 3] = 255
                img = img_copy

        return img

    def _numpy_to_qimage_scaled(self, img_array):
        """Convert RGBA numpy array to QImage, scaled to fit MAX_DISPLAY_PX."""
        h, w = img_array.shape[:2]
        # Ensure contiguous
        img_array = np.ascontiguousarray(img_array)
        qimg = QImage(img_array.data, w, h, w * 4, QImage.Format_RGBA8888).copy()

        # Scale to fit display
        max_dim = max(h, w)
        if max_dim > 0:
            scale = self.MAX_DISPLAY_PX / max_dim
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            qimg = qimg.scaled(new_w, new_h)

        return qimg
