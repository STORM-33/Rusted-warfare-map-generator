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

const getCcTeamColor = (
  unitsTileset: ExtractedTileset | undefined,
  localId: number,
): string => {
  if (unitsTileset && localId >= 0 && localId < unitsTileset.tileCount) {
    const offset = localId * 4;
    const r = unitsTileset.centerLut[offset];
    const g = unitsTileset.centerLut[offset + 1];
    const b = unitsTileset.centerLut[offset + 2];
    return `rgba(${r},${g},${b},0.94)`;
  }
  return "rgba(50,100,255,0.94)";
};

const getCcPlayerLabel = (localId: number): string => {
  const pair = localId % 5;
  const team = localId < 5 ? 0 : 1;
  return String(pair * 2 + team + 1);
};

export const renderOverlay = (
  canvas: HTMLCanvasElement,
  snapshot: WizardSnapshot,
  mode: RenderMode,
  tilesets?: ExtractedTilesets,
  tileSize = 20,
) => {
  const { rows, cols } = getSnapshotDimensions(snapshot);
  if (rows === 0 || cols === 0) {
    canvas.width = 1;
    canvas.height = 1;
    return;
  }

  // In sampled mode, scale the overlay up so CC/resource markers and labels
  // have enough pixel resolution (mirroring the EXE which scales the overlay
  // pixmap to the widget size before drawing text).
  const cellSize = mode === "full" ? tileSize : Math.max(1, Math.ceil(600 / Math.max(rows, cols)));
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
          wallValue === 2 ? "rgba(50,150,200,0.55)" : "rgba(200,50,50,0.55)";
        context.fillRect(col * cellSize, row * cellSize, cellSize, cellSize);
      }
    }
  }

  const unitsMatrix = snapshot.matrices.units_matrix;
  const unitsTileset = tilesets?.units;
  const unitsFirstGid = unitsTileset?.firstGid ?? 101;

  // CC labels collected for text pass (drawn after all fills so text is on top)
  const ccLabels: { label: string; cx: number; cy: number }[] = [];

  for (const [row, col] of snapshot.cc_positions) {
    const gid = unitsMatrix ? getMatrixValue(unitsMatrix, row, col) : 0;
    const localId = gid > 0 ? gid - unitsFirstGid : 0;
    const x = col * cellSize;
    const y = row * cellSize;

    // Dark border (7x7 cells) and colored fill (5x5 cells), matching EXE overlay
    context.fillStyle = "rgba(0,0,0,0.78)";
    context.fillRect(x - 3 * cellSize, y - 3 * cellSize, 7 * cellSize, 7 * cellSize);
    context.fillStyle = getCcTeamColor(unitsTileset, localId);
    context.fillRect(x - 2 * cellSize, y - 2 * cellSize, 5 * cellSize, 5 * cellSize);
    ccLabels.push({ label: getCcPlayerLabel(localId), cx: x, cy: y });
  }

  // Resource pools: golden yellow 3x3 cells, matching EXE overlay
  for (const [row, col] of snapshot.resource_positions) {
    const x = col * cellSize;
    const y = row * cellSize;
    context.fillStyle = "rgba(255,220,50,0.86)";
    context.fillRect(x - 1 * cellSize, y - 1 * cellSize, 3 * cellSize, 3 * cellSize);
  }

  // Draw CC player labels on top
  if (ccLabels.length > 0) {
    const fontSize = Math.max(8, Math.floor(cellSize * 3));
    context.font = `bold ${fontSize}px Arial, sans-serif`;
    context.textAlign = "center";
    context.textBaseline = "middle";
    for (const { label, cx, cy } of ccLabels) {
      // Black outline
      context.strokeStyle = "rgba(0,0,0,0.86)";
      context.lineWidth = 2;
      context.strokeText(label, cx, cy);
      // White foreground
      context.fillStyle = "rgba(255,255,255,1)";
      context.fillText(label, cx, cy);
    }
  }
};
