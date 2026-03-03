"use client";

import { useCallback, useEffect, useMemo, useRef } from "react";
import type { HillDrawingMode, Polygon, RenderMode, WizardSnapshot } from "@/lib/types";
import type { ExtractedTilesets } from "@/lib/tilesetExtractor";
import { renderOverlay, renderSnapshotBase } from "@/lib/canvasRenderer";

type InteractionMode = "none" | "draw" | "click" | "polygon";
type RenderPreference = RenderMode | "auto";

// Depth colors for polygon mode visualization
const DEPTH_COLORS = [
  "transparent",
  "rgba(16, 185, 129, 0.25)",   // 1
  "rgba(16, 185, 129, 0.35)",   // 2
  "rgba(16, 185, 129, 0.45)",   // 3
  "rgba(16, 185, 129, 0.55)",   // 4
  "rgba(16, 185, 129, 0.65)",   // 5
  "rgba(16, 185, 129, 0.75)",   // 6
  "rgba(16, 185, 129, 0.85)",   // 7
  "rgba(16, 185, 129, 0.90)",   // 8
  "rgba(16, 185, 129, 0.95)",   // 9
];

type MapCanvasProps = {
  snapshot: WizardSnapshot | null;
  tilesets?: ExtractedTilesets;
  view: "coastline" | "height" | "final";
  requestedMode: RenderPreference;
  interactionMode: InteractionMode;
  drawValue: 1 | 2;
  eraseMode?: boolean;
  canvasIdPrefix?: string;
  polygons?: Polygon[];
  drawingPolygon?: Polygon | null;
  selectedPolygonId?: number | null;
  selectedEdgeIndex?: number | null;
  onDraw?: (points: [number, number][], value: 0 | 1 | 2) => void;
  onClickCell?: (row: number, col: number, isRightClick: boolean) => void;
  onPolygonClick?: (row: number, col: number) => void;
  onPolygonRightClick?: () => void;
  onRenderModeChange?: (mode: RenderMode) => void;
  hillMode?: HillDrawingMode | null;
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
  eraseMode = false,
  canvasIdPrefix = "map-canvas",
  polygons = [],
  drawingPolygon = null,
  selectedPolygonId = null,
  selectedEdgeIndex = null,
  onDraw,
  onClickCell,
  onPolygonClick,
  onPolygonRightClick,
  onRenderModeChange,
  hillMode,
}: MapCanvasProps) {
  const baseCanvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const polygonCanvasRef = useRef<HTMLCanvasElement>(null);
  const activeRenderModeRef = useRef<RenderMode>("sampled");
  const drawingRef = useRef(false);
  const lastCellRef = useRef<string>("");
  const cursorCellRef = useRef<{ row: number; col: number } | null>(null);

  const dimensions = useMemo(() => getDimensions(snapshot), [snapshot]);

  // Base rendering
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

  // Render depth-based overlay for polygon mode
  const renderDepthOverlay = useCallback((ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement) => {
    if (!snapshot?.matrices.wall_matrix) return;
    
    const { shape, data } = snapshot.matrices.wall_matrix;
    const [rows, cols] = shape;
    const cellSize = activeRenderModeRef.current === "full" ? 20 : 1;
    
    // Render depth values as colored cells
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const depth = data[r * cols + c];
        if (depth > 0 && depth < DEPTH_COLORS.length) {
          ctx.fillStyle = DEPTH_COLORS[depth];
          ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
        }
      }
    }
  }, [snapshot]);

  // Polygon overlay rendering
  const renderPolygonOverlay = useCallback(() => {
    const canvas = polygonCanvasRef.current;
    const baseCanvas = baseCanvasRef.current;
    if (!canvas || !baseCanvas) return;
    canvas.width = baseCanvas.width;
    canvas.height = baseCanvas.height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const cellSize = activeRenderModeRef.current === "full" ? 20 : 1;

    const toX = (col: number) => col * cellSize + cellSize / 2;
    const toY = (row: number) => row * cellSize + cellSize / 2;

    // If in polygon mode, render depth overlay first
    if (hillMode === "polygon") {
      renderDepthOverlay(ctx, canvas);
    }

    const drawPoly = (poly: Polygon, isSelected: boolean, isDrawing: boolean) => {
      const verts = poly.vertices;
      if (verts.length === 0) return;

      // Fill for closed polygons (semi-transparent, but depth overlay is already showing)
      if (poly.closed && verts.length >= 3) {
        ctx.beginPath();
        ctx.moveTo(toX(verts[0][1]), toY(verts[0][0]));
        for (let i = 1; i < verts.length; i++) {
          ctx.lineTo(toX(verts[i][1]), toY(verts[i][0]));
        }
        ctx.closePath();
        // Just a subtle outline fill
        ctx.fillStyle = isSelected
          ? "rgba(56, 189, 248, 0.1)"
          : "rgba(16, 185, 129, 0.05)";
        ctx.fill();
      }

      // Edges
      const edgeColor = isSelected ? "#38bdf8" : "#10b981";
      const gapColor = isSelected ? "rgba(56, 189, 248, 0.5)" : "rgba(16, 185, 129, 0.5)";
      
      for (let i = 0; i < verts.length; i++) {
        const nextIdx = (i + 1) % verts.length;
        if (!poly.closed && i === verts.length - 1) break;
        
        const isGap = i < poly.edgeGaps.length && poly.edgeGaps[i];
        const isSelectedEdge = isSelected && selectedEdgeIndex === i;
        
        ctx.beginPath();
        ctx.moveTo(toX(verts[i][1]), toY(verts[i][0]));
        ctx.lineTo(toX(verts[nextIdx][1]), toY(verts[nextIdx][0]));
        
        ctx.lineWidth = isSelectedEdge ? Math.max(2, cellSize / 3) : Math.max(1, cellSize / 6);
        
        if (isGap) {
          ctx.setLineDash([cellSize * 0.4, cellSize * 0.4]);
          ctx.strokeStyle = gapColor;
        } else {
          ctx.setLineDash([]);
          ctx.strokeStyle = edgeColor;
        }
        
        ctx.stroke();
        
        // Highlight selected edge
        if (isSelectedEdge) {
          ctx.setLineDash([]);
          ctx.strokeStyle = "#fbbf24"; // Amber highlight
          ctx.lineWidth = Math.max(3, cellSize / 4);
          ctx.stroke();
        }
      }
      ctx.setLineDash([]);

      // Vertices
      for (let i = 0; i < verts.length; i++) {
        const [r, c] = verts[i];
        ctx.beginPath();
        ctx.arc(toX(c), toY(r), Math.max(2, cellSize / 3), 0, 2 * Math.PI);
        ctx.fillStyle = i === 0 && isDrawing ? "#fbbf24" : isSelected ? "#38bdf8" : "#10b981";
        ctx.fill();
      }
    };

    // Draw all polygons
    for (const poly of polygons) {
      drawPoly(poly, poly.id === selectedPolygonId, false);
    }

    // Draw in-progress polygon
    if (drawingPolygon && drawingPolygon.vertices.length > 0) {
      drawPoly(drawingPolygon, false, true);

      // Preview line from last vertex to cursor
      const cursor = cursorCellRef.current;
      if (cursor) {
        const lastVert = drawingPolygon.vertices[drawingPolygon.vertices.length - 1];
        ctx.beginPath();
        ctx.moveTo(toX(lastVert[1]), toY(lastVert[0]));
        ctx.lineTo(toX(cursor.col), toY(cursor.row));
        ctx.setLineDash([cellSize * 0.3, cellSize * 0.3]);
        ctx.strokeStyle = "rgba(251, 191, 36, 0.6)";
        ctx.lineWidth = Math.max(1, cellSize / 6);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }
  }, [polygons, drawingPolygon, selectedPolygonId, selectedEdgeIndex, hillMode, renderDepthOverlay]);

  useEffect(() => {
    renderPolygonOverlay();
  }, [renderPolygonOverlay]);

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
    if (interactionMode === "polygon") {
      if (event.button === 2) {
        event.preventDefault();
        onPolygonRightClick?.();
      } else if (event.button === 0) {
        onPolygonClick?.(cell.row, cell.col);
      }
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
    const value: 0 | 1 | 2 = event.button === 2 || eraseMode ? 0 : drawValue;
    onDraw?.([[cell.row, cell.col]], value);
    lastCellRef.current = `${cell.row}:${cell.col}:${value}`;
  };

  const onPointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    // Update cursor position for polygon preview line
    if (interactionMode === "polygon") {
      const cell = eventToCell(event);
      cursorCellRef.current = cell;
      if (drawingPolygon && drawingPolygon.vertices.length > 0) {
        renderPolygonOverlay();
      }
    }

    if (!drawingRef.current || interactionMode !== "draw") {
      return;
    }
    const cell = eventToCell(event);
    if (!cell) {
      return;
    }
    const value: 0 | 1 | 2 = event.buttons & 2 || eraseMode ? 0 : drawValue;
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
    ? `min(100%, calc((100vh - 14rem) * ${dimensions.cols} / ${dimensions.rows}))`
    : "min(100%, calc(100vh - 14rem))";

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
      <canvas id={`${canvasIdPrefix}-polygon`} ref={polygonCanvasRef} className="map-layer overlay" />
    </div>
  );
}
