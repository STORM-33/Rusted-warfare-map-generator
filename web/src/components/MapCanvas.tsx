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
  brushSize: number;
  onDraw?: (points: [number, number][], value: 0 | 1 | 2) => void;
  onClickCell?: (row: number, col: number) => void;
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
  brushSize,
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

  const onPointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    const cell = eventToCell(event);
    if (!cell) {
      return;
    }
    if (interactionMode === "click" && event.button === 0) {
      onClickCell?.(cell.row, cell.col);
      return;
    }
    if (interactionMode !== "draw") {
      return;
    }
    event.preventDefault();
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
    lastCellRef.current = marker;
    onDraw?.([[cell.row, cell.col]], value);
  };

  const stopDrawing = () => {
    drawingRef.current = false;
    lastCellRef.current = "";
  };

  return (
    <div
      className={`map-canvas ${interactionMode}`}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={stopDrawing}
      onPointerLeave={stopDrawing}
      onContextMenu={(event) => event.preventDefault()}
    >
      <canvas ref={baseCanvasRef} className="map-layer" />
      <canvas ref={overlayCanvasRef} className="map-layer overlay" />
    </div>
  );
}
