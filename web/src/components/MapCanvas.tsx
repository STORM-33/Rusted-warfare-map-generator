"use client";

import { useEffect, useMemo, useRef } from "react";
import type { RenderMode, WizardSnapshot } from "@/lib/types";
import type { ExtractedTilesets } from "@/lib/tilesetExtractor";
import { renderOverlay, renderSnapshotBase } from "@/lib/canvasRenderer";

type InteractionMode = "none" | "draw" | "click";
type RenderPreference = RenderMode | "auto";

type MapCanvasProps = {
  snapshot: WizardSnapshot | null;
  tilesets?: ExtractedTilesets;
  view: "coastline" | "height" | "final";
  requestedMode: RenderPreference;
  interactionMode: InteractionMode;
  drawValue: 1 | 2;
  canvasIdPrefix?: string;
  onDraw?: (points: [number, number][], value: 0 | 1 | 2) => void;
  onClickCell?: (row: number, col: number, isRightClick: boolean) => void;
  onRenderModeChange?: (mode: RenderMode) => void;
};

const getDimensions = (snapshot: WizardSnapshot | null) => {
  if (!snapshot) {
    return null;
  }
  const source =
    snapshot.matrices.id_matrix ??
    snapshot.matrices.height_map ??
    snapshot.matrices.coastline_height_map ??
    snapshot.matrices.wall_matrix;
  if (!source) {
    return null;
  }
  return { rows: source.shape[0], cols: source.shape[1] };
};

export function MapCanvas({
  snapshot,
  tilesets,
  view,
  requestedMode,
  interactionMode,
  drawValue,
  canvasIdPrefix = "map-canvas",
  onDraw,
  onClickCell,
  onRenderModeChange,
}: MapCanvasProps) {
  const baseCanvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const activeRenderModeRef = useRef<RenderMode>("sampled");
  const drawingRef = useRef(false);
  const lastCellRef = useRef<string>("");

  const dimensions = useMemo(() => getDimensions(snapshot), [snapshot]);

  useEffect(() => {
    const baseCanvas = baseCanvasRef.current;
    const overlayCanvas = overlayCanvasRef.current;
    if (!baseCanvas || !overlayCanvas) {
      return;
    }
    if (!snapshot) {
      baseCanvas.width = 1;
      baseCanvas.height = 1;
      overlayCanvas.width = 1;
      overlayCanvas.height = 1;
      return;
    }
    const mode = renderSnapshotBase(
      baseCanvas,
      snapshot,
      tilesets,
      view,
      requestedMode,
    );
    activeRenderModeRef.current = mode;
    onRenderModeChange?.(mode);
    renderOverlay(overlayCanvas, snapshot, mode, tilesets);
  }, [onRenderModeChange, requestedMode, snapshot, tilesets, view]);

  const eventToCell = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!dimensions) {
      return null;
    }
    const baseCanvas = baseCanvasRef.current;
    if (!baseCanvas) {
      return null;
    }
    const rect = baseCanvas.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) {
      return null;
    }
    const localX = ((event.clientX - rect.left) / rect.width) * baseCanvas.width;
    const localY = ((event.clientY - rect.top) / rect.height) * baseCanvas.height;
    const cellSize = activeRenderModeRef.current === "full" ? 20 : 1;
    const col = Math.max(0, Math.min(dimensions.cols - 1, Math.floor(localX / cellSize)));
    const row = Math.max(0, Math.min(dimensions.rows - 1, Math.floor(localY / cellSize)));
    return { row, col };
  };

  const interpolateCells = (
    r0: number,
    c0: number,
    r1: number,
    c1: number,
  ): [number, number][] => {
    const dr = Math.abs(r1 - r0);
    const dc = Math.abs(c1 - c0);
    const sr = r0 < r1 ? 1 : -1;
    const sc = c0 < c1 ? 1 : -1;
    let err = dr - dc;
    let r = r0;
    let c = c0;
    const cells: [number, number][] = [];
    for (; ;) {
      cells.push([r, c]);
      if (r === r1 && c === c1) {
        break;
      }
      const e2 = 2 * err;
      if (e2 > -dc) {
        err -= dc;
        r += sr;
      }
      if (e2 < dr) {
        err += dr;
        c += sc;
      }
    }
    return cells;
  };

  const onPointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    const cell = eventToCell(event);
    if (!cell) {
      return;
    }
    if (interactionMode === "click" && (event.button === 0 || event.button === 2)) {
      onClickCell?.(cell.row, cell.col, event.button === 2);
      return;
    }
    if (interactionMode !== "draw") {
      return;
    }
    event.preventDefault();
    (event.currentTarget as HTMLDivElement).setPointerCapture(event.pointerId);
    drawingRef.current = true;
    const value: 0 | 1 | 2 = event.button === 2 ? 0 : drawValue;
    onDraw?.([[cell.row, cell.col]], value);
    lastCellRef.current = `${cell.row}:${cell.col}:${value}`;
  };

  const onPointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!drawingRef.current || interactionMode !== "draw") {
      return;
    }
    const cell = eventToCell(event);
    if (!cell) {
      return;
    }
    const value: 0 | 1 | 2 = event.buttons & 2 ? 0 : drawValue;
    const marker = `${cell.row}:${cell.col}:${value}`;
    if (marker === lastCellRef.current) {
      return;
    }
    // Interpolate from last cell to current cell to fill gaps from fast movement
    const lastParts = lastCellRef.current.split(":");
    lastCellRef.current = marker;
    if (lastParts.length >= 2) {
      const prevRow = Number(lastParts[0]);
      const prevCol = Number(lastParts[1]);
      if (Number.isFinite(prevRow) && Number.isFinite(prevCol)) {
        const cells = interpolateCells(prevRow, prevCol, cell.row, cell.col);
        // Skip the first cell — it was already drawn in the previous event
        if (cells.length > 1) {
          onDraw?.(cells.slice(1), value);
          return;
        }
      }
    }
    onDraw?.([[cell.row, cell.col]], value);
  };

  const stopDrawing = (event: React.PointerEvent<HTMLDivElement>) => {
    if (drawingRef.current) {
      (event.currentTarget as HTMLDivElement).releasePointerCapture(event.pointerId);
    }
    drawingRef.current = false;
    lastCellRef.current = "";
  };

  const widthExpr = dimensions
    ? `min(100%, calc((100vh - 10rem) * ${dimensions.cols} / ${dimensions.rows}))`
    : "min(100%, calc(100vh - 10rem))";

  return (
    <div
      className={`map-canvas ${interactionMode}`}
      style={{ width: widthExpr }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={stopDrawing}
      onContextMenu={(event) => event.preventDefault()}
    >
      <canvas id={`${canvasIdPrefix}-base`} ref={baseCanvasRef} className="map-layer" />
      <canvas id={`${canvasIdPrefix}-overlay`} ref={overlayCanvasRef} className="map-layer overlay" />
    </div>
  );
}
