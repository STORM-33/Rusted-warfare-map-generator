import type { MatrixPayload, RenderMode, WizardSnapshot } from "@/lib/types";
import type { ExtractedTileset, ExtractedTilesets } from "@/lib/tilesetExtractor";

type RequestedRenderMode = RenderMode | "auto";

const LEVEL_FLAT_TILES: Record<number, number> = {
  [-2]: 83,
  [-1]: 28,
  0: 31,
  1: 34,
  2: 37,
  3: 40,
  4: 43,
  5: 46,
  6: 49,
  7: 52,
};

const FALLBACK_LEVEL_COLORS: Record<number, [number, number, number, number]> = {
  [-2]: [11, 30, 71, 255],
  [-1]: [19, 56, 112, 255],
  0: [36, 88, 148, 255],
  1: [194, 172, 108, 255],
  2: [97, 157, 84, 255],
  3: [124, 108, 84, 255],
  4: [95, 98, 80, 255],
  5: [128, 128, 136, 255],
  6: [219, 224, 235, 255],
  7: [236, 248, 255, 255],
};

const getMatrixValue = (matrix: MatrixPayload, row: number, col: number) => {
  const cols = matrix.shape[1];
  return matrix.data[row * cols + col] ?? 0;
};

const getLutColor = (
  tileset: ExtractedTileset | undefined,
  localId: number,
  fallback: [number, number, number, number] = [0, 0, 0, 0],
): [number, number, number, number] => {
  if (!tileset || localId < 0 || localId >= tileset.tileCount) {
    return fallback;
  }
  const offset = localId * 4;
  return [
    tileset.centerLut[offset],
    tileset.centerLut[offset + 1],
    tileset.centerLut[offset + 2],
    tileset.centerLut[offset + 3],
  ];
};

const blendPixel = (
  pixels: Uint8ClampedArray,
  offset: number,
  rgba: [number, number, number, number],
) => {
  const alpha = (rgba[3] ?? 255) / 255;
  const inv = 1 - alpha;
  pixels[offset] = Math.round(rgba[0] * alpha + pixels[offset] * inv);
  pixels[offset + 1] = Math.round(rgba[1] * alpha + pixels[offset + 1] * inv);
  pixels[offset + 2] = Math.round(rgba[2] * alpha + pixels[offset + 2] * inv);
  pixels[offset + 3] = 255;
};

const renderHeightMap = (
  canvas: HTMLCanvasElement,
  matrix: MatrixPayload,
  tilesets?: ExtractedTilesets,
) => {
  const [rows, cols] = matrix.shape;
  canvas.width = cols;
  canvas.height = rows;
  const context = canvas.getContext("2d");
  if (!context) {
    return;
  }

  const pixels = new Uint8ClampedArray(rows * cols * 4);
  const levelColors = new Map<number, [number, number, number, number]>();

  for (const [levelKey, tileId] of Object.entries(LEVEL_FLAT_TILES)) {
    const level = Number(levelKey);
    const fallback = FALLBACK_LEVEL_COLORS[level] ?? [120, 120, 120, 255];
    levelColors.set(level, getLutColor(tilesets?.ground, tileId, fallback));
  }

  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const level = getMatrixValue(matrix, row, col);
      const color =
        levelColors.get(level) ??
        ([120, 120, 120, 255] as [number, number, number, number]);
      const offset = (row * cols + col) * 4;
      pixels[offset] = color[0];
      pixels[offset + 1] = color[1];
      pixels[offset + 2] = color[2];
      pixels[offset + 3] = color[3];
    }
  }

  context.putImageData(new ImageData(pixels, cols, rows), 0, 0);
};

const drawTile = (
  context: CanvasRenderingContext2D,
  tileset: ExtractedTileset | undefined,
  localId: number,
  dx: number,
  dy: number,
) => {
  if (!tileset || localId < 0 || localId >= tileset.tileCount) {
    return;
  }
  const sx = (localId % tileset.columns) * tileset.tileWidth;
  const sy = Math.floor(localId / tileset.columns) * tileset.tileHeight;
  context.drawImage(
    tileset.image,
    sx,
    sy,
    tileset.tileWidth,
    tileset.tileHeight,
    dx,
    dy,
    tileset.tileWidth,
    tileset.tileHeight,
  );
};

const renderTerrainSampled = (
  canvas: HTMLCanvasElement,
  idMatrix: MatrixPayload,
  snapshot: WizardSnapshot,
  tilesets?: ExtractedTilesets,
) => {
  const [rows, cols] = idMatrix.shape;
  canvas.width = cols;
  canvas.height = rows;
  const context = canvas.getContext("2d");
  if (!context) {
    return;
  }

  const itemsMatrix = snapshot.matrices.items_matrix;
  const unitsMatrix = snapshot.matrices.units_matrix;
  const pixels = new Uint8ClampedArray(rows * cols * 4);

  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const id = getMatrixValue(idMatrix, row, col);
      const color = getLutColor(tilesets?.ground, id, [90, 90, 90, 255]);
      const offset = (row * cols + col) * 4;
      pixels[offset] = color[0];
      pixels[offset + 1] = color[1];
      pixels[offset + 2] = color[2];
      pixels[offset + 3] = color[3];

      if (itemsMatrix) {
        const itemGid = getMatrixValue(itemsMatrix, row, col);
        if (itemGid > 0) {
          let itemColor: [number, number, number, number] = [0, 0, 0, 0];
          if (tilesets?.wall && itemGid >= tilesets.wall.firstGid) {
            itemColor = getLutColor(
              tilesets.wall,
              itemGid - tilesets.wall.firstGid,
              itemColor,
            );
          } else if (tilesets?.items) {
            itemColor = getLutColor(
              tilesets.items,
              itemGid - tilesets.items.firstGid,
              itemColor,
            );
          }
          if (itemColor[3] > 0) {
            blendPixel(pixels, offset, itemColor);
          }
        }
      }

      if (unitsMatrix && tilesets?.units) {
        const unitGid = getMatrixValue(unitsMatrix, row, col);
        if (unitGid >= tilesets.units.firstGid) {
          const unitColor = getLutColor(
            tilesets.units,
            unitGid - tilesets.units.firstGid,
            [0, 0, 0, 0],
          );
          if (unitColor[3] > 0) {
            blendPixel(pixels, offset, unitColor);
          }
        }
      }
    }
  }

  context.putImageData(new ImageData(pixels, cols, rows), 0, 0);
};

const renderTerrainFull = (
  canvas: HTMLCanvasElement,
  idMatrix: MatrixPayload,
  snapshot: WizardSnapshot,
  tilesets: ExtractedTilesets,
) => {
  const [rows, cols] = idMatrix.shape;
  const tileWidth = tilesets.ground?.tileWidth ?? 20;
  const tileHeight = tilesets.ground?.tileHeight ?? 20;

  canvas.width = cols * tileWidth;
  canvas.height = rows * tileHeight;

  const context = canvas.getContext("2d");
  if (!context) {
    return;
  }
  context.imageSmoothingEnabled = false;
  context.clearRect(0, 0, canvas.width, canvas.height);

  const itemsMatrix = snapshot.matrices.items_matrix;
  const unitsMatrix = snapshot.matrices.units_matrix;

  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const dx = col * tileWidth;
      const dy = row * tileHeight;
      drawTile(context, tilesets.ground, getMatrixValue(idMatrix, row, col), dx, dy);
      if (itemsMatrix) {
        const itemGid = getMatrixValue(itemsMatrix, row, col);
        if (itemGid > 0) {
          if (tilesets.wall && itemGid >= tilesets.wall.firstGid) {
            drawTile(context, tilesets.wall, itemGid - tilesets.wall.firstGid, dx, dy);
          } else if (tilesets.items) {
            drawTile(
              context,
              tilesets.items,
              itemGid - tilesets.items.firstGid,
              dx,
              dy,
            );
          }
        }
      }
      if (unitsMatrix && tilesets.units) {
        const unitGid = getMatrixValue(unitsMatrix, row, col);
        if (unitGid >= tilesets.units.firstGid) {
          drawTile(
            context,
            tilesets.units,
            unitGid - tilesets.units.firstGid,
            dx,
            dy,
          );
        }
      }
    }
  }
};

const getSnapshotDimensions = (snapshot: WizardSnapshot) => {
  const candidate =
    snapshot.matrices.id_matrix ??
    snapshot.matrices.height_map ??
    snapshot.matrices.coastline_height_map ??
    snapshot.matrices.wall_matrix;
  if (!candidate) {
    return { rows: 0, cols: 0 };
  }
  return { rows: candidate.shape[0], cols: candidate.shape[1] };
};

export const resolveRenderMode = (
  rows: number,
  cols: number,
  requested: RequestedRenderMode,
  maxFullSize = 384,
): RenderMode => {
  if (requested === "sampled" || requested === "full") {
    return requested;
  }
  return Math.max(rows, cols) <= maxFullSize ? "full" : "sampled";
};

export const renderSnapshotBase = (
  canvas: HTMLCanvasElement,
  snapshot: WizardSnapshot,
  tilesets: ExtractedTilesets | undefined,
  view: "coastline" | "height" | "final",
  requestedMode: RequestedRenderMode = "auto",
) => {
  const { rows, cols } = getSnapshotDimensions(snapshot);
  if (rows === 0 || cols === 0) {
    return "sampled" as RenderMode;
  }

  if (view !== "final") {
    const map =
      view === "coastline"
        ? snapshot.matrices.coastline_height_map
        : snapshot.matrices.height_map ?? snapshot.matrices.coastline_height_map;
    if (map) {
      renderHeightMap(canvas, map, tilesets);
    }
    return "sampled" as RenderMode;
  }

  const idMatrix = snapshot.matrices.id_matrix ?? snapshot.matrices.height_map;
  if (!idMatrix) {
    return "sampled" as RenderMode;
  }

  const mode = resolveRenderMode(rows, cols, requestedMode);
  if (mode === "full" && tilesets?.ground) {
    renderTerrainFull(canvas, idMatrix, snapshot, tilesets);
  } else {
    renderTerrainSampled(canvas, idMatrix, snapshot, tilesets);
  }
  return mode;
};

export const renderOverlay = (
  canvas: HTMLCanvasElement,
  snapshot: WizardSnapshot,
  mode: RenderMode,
  tileSize = 20,
) => {
  const { rows, cols } = getSnapshotDimensions(snapshot);
  if (rows === 0 || cols === 0) {
    canvas.width = 1;
    canvas.height = 1;
    return;
  }

  const cellSize = mode === "full" ? tileSize : 1;
  canvas.width = cols * cellSize;
  canvas.height = rows * cellSize;

  const context = canvas.getContext("2d");
  if (!context) {
    return;
  }
  context.clearRect(0, 0, canvas.width, canvas.height);

  const wallMatrix = snapshot.matrices.wall_matrix;
  if (wallMatrix) {
    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) {
        const wallValue = getMatrixValue(wallMatrix, row, col);
        if (wallValue === 0) {
          continue;
        }
        context.fillStyle =
          wallValue === 2 ? "rgba(59,130,246,0.55)" : "rgba(249,115,22,0.45)";
        context.fillRect(col * cellSize, row * cellSize, cellSize, cellSize);
      }
    }
  }

  const unitsMatrix = snapshot.matrices.units_matrix;
  for (const [row, col] of snapshot.cc_positions) {
    const gid = unitsMatrix ? getMatrixValue(unitsMatrix, row, col) : 0;
    const isTeamB = gid >= 106;
    context.fillStyle = isTeamB ? "rgba(59,130,246,0.95)" : "rgba(239,68,68,0.95)";
    const x = col * cellSize;
    const y = row * cellSize;
    if (cellSize === 1) {
      context.fillRect(x, y, 1, 1);
    } else {
      context.beginPath();
      context.arc(
        x + cellSize / 2,
        y + cellSize / 2,
        Math.max(2, cellSize * 0.28),
        0,
        Math.PI * 2,
      );
      context.fill();
    }
  }

  context.fillStyle = "rgba(34,197,94,0.95)";
  for (const [row, col] of snapshot.resource_positions) {
    const x = col * cellSize;
    const y = row * cellSize;
    if (cellSize === 1) {
      context.fillRect(x, y, 1, 1);
    } else {
      context.fillRect(
        x + Math.floor(cellSize * 0.25),
        y + Math.floor(cellSize * 0.25),
        Math.max(2, Math.floor(cellSize * 0.5)),
        Math.max(2, Math.floor(cellSize * 0.5)),
      );
    }
  }
};
