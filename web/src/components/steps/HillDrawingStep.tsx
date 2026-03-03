"use client";

import type { HillDrawingMode, Polygon } from "@/lib/types";

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

type HillDrawingStepProps = {
  // Mode
  mode: HillDrawingMode;
  onModeChange: (mode: HillDrawingMode) => void;
  // Brush mode
  drawValue: 1 | 2;
  eraseMode: boolean;
  brushSize: number;
  onDrawValueChange: (value: 1 | 2) => void;
  onEraseModeChange: (erasing: boolean) => void;
  onBrushSizeChange: (size: number) => void;
  onClearBrush: () => void;
  onUndoBrush: () => void;
  onRedoBrush: () => void;
  // Polygon mode
  polygons: Polygon[];
  drawingPolygon: Polygon | null;
  selectedPolygonId: number | null;
  selectedEdgeIndex: number | null;
  onPolygonClick?: (row: number, col: number) => void;
  onToggleEdgeGap: (polygonId: number, edgeIndex: number) => void;
  onDeletePolygon: () => void;
  onClearPolygons: () => void;
  onCancelDrawing: () => void;
  onUndoPolygons: () => void;
  onRedoPolygons: () => void;
  // Common
  disabled?: boolean;
};

export function HillDrawingStep({
  mode,
  onModeChange,
  drawValue,
  eraseMode,
  brushSize,
  onDrawValueChange,
  onEraseModeChange,
  onBrushSizeChange,
  onClearBrush,
  onUndoBrush,
  onRedoBrush,
  polygons,
  drawingPolygon,
  selectedPolygonId,
  selectedEdgeIndex,
  onToggleEdgeGap,
  onDeletePolygon,
  onClearPolygons,
  onCancelDrawing,
  onUndoPolygons,
  onRedoPolygons,
  disabled,
}: HillDrawingStepProps) {
  const closedPolygons = polygons.filter((p) => p.closed);
  const selectedPolygon = polygons.find((p) => p.id === selectedPolygonId);

  return (
    <section className="panel-section">
      <h2>2. Hill Drawing</h2>

      <div className="hill-mode-tabs">
        <button
          type="button"
          className={`mode-tab ${mode === "brush" ? "active" : ""}`}
          onClick={() => onModeChange("brush")}
          disabled={disabled}
        >
          Brush
        </button>
        <button
          type="button"
          className={`mode-tab ${mode === "polygon" ? "active" : ""}`}
          onClick={() => onModeChange("polygon")}
          disabled={disabled}
        >
          Polygon Layers
        </button>
      </div>

      {mode === "brush" ? (
        <>
          <p>Draw on the map canvas. Use eraser or right-click to clear cells.</p>
          
          <div className="button-row">
            <button
              type="button"
              className={drawValue === 1 && !eraseMode ? "active-btn" : "ghost-btn"}
              onClick={() => onDrawValueChange(1)}
              disabled={disabled}
            >
              Wall Brush
            </button>
            <button
              type="button"
              className={drawValue === 2 && !eraseMode ? "active-btn" : "ghost-btn"}
              onClick={() => onDrawValueChange(2)}
              disabled={disabled}
            >
              Gap Brush
            </button>
            <button
              type="button"
              className={eraseMode ? "active-btn" : "ghost-btn"}
              onClick={() => onEraseModeChange(!eraseMode)}
              disabled={disabled}
            >
              Eraser
            </button>
            <button type="button" className="ghost-btn" onClick={onClearBrush} disabled={disabled}>
              Clear
            </button>
          </div>
          
          <div className="button-row">
            <button type="button" className="ghost-btn" onClick={onUndoBrush} disabled={disabled}>
              Undo
            </button>
            <button type="button" className="ghost-btn" onClick={onRedoBrush} disabled={disabled}>
              Redo
            </button>
          </div>
          
          <label>
            Brush size: {brushSize}
            <input
              type="range"
              min={1}
              max={9}
              step={2}
              value={brushSize}
              onChange={(e) => onBrushSizeChange(Number(e.target.value))}
              disabled={disabled}
            />
          </label>
        </>
      ) : (
        <>
          <p className="polygon-status">
            {drawingPolygon
              ? `Drawing polygon — ${drawingPolygon.vertices.length} vertices placed. ${
                  drawingPolygon.vertices.length >= 3
                    ? "Click first vertex to close."
                    : "Click to add vertices."
                }`
              : `${closedPolygons.length} polygon${closedPolygons.length !== 1 ? "s" : ""} placed`
            }
          </p>
          
          <div className="button-row">
            {drawingPolygon && (
              <button
                type="button"
                className="ghost-btn"
                onClick={onCancelDrawing}
                disabled={disabled}
              >
                Cancel Drawing
              </button>
            )}
            {selectedPolygon && !drawingPolygon && (
              <>
                <button
                  type="button"
                  className="ghost-btn"
                  onClick={onDeletePolygon}
                  disabled={disabled}
                >
                  Delete Polygon
                </button>
                {selectedEdgeIndex !== null && (
                  <button
                    type="button"
                    className="ghost-btn"
                    onClick={() => onToggleEdgeGap(selectedPolygonId!, selectedEdgeIndex)}
                    disabled={disabled}
                  >
                    Toggle Gap
                  </button>
                )}
              </>
            )}
            {closedPolygons.length > 0 && !drawingPolygon && (
              <button
                type="button"
                className="ghost-btn"
                onClick={onClearPolygons}
                disabled={disabled}
              >
                Clear All
              </button>
            )}
          </div>
          
          <div className="button-row">
            <button type="button" className="ghost-btn" onClick={onUndoPolygons} disabled={disabled}>
              Undo
            </button>
            <button type="button" className="ghost-btn" onClick={onRedoPolygons} disabled={disabled}>
              Redo
            </button>
          </div>

          {/* Legend */}
          <div className="polygon-legend">
            <h4>Elevation Levels</h4>
            <div className="legend-grid">
              {[1, 3, 5, 7, 9].map((level) => (
                <div key={level} className="legend-item">
                  <span
                    className="depth-swatch"
                    style={{ background: DEPTH_COLORS[level] }}
                  />
                  <span>Level {level}</span>
                </div>
              ))}
            </div>
            <div className="legend-hints">
              <div className="hint">
                <span className="line-example solid" />
                <span>Solid = Wall</span>
              </div>
              <div className="hint">
                <span className="line-example dashed" />
                <span>Dashed = Gap</span>
              </div>
            </div>
            <p className="tip">
              <strong>Tip:</strong> Nested polygons create elevation. 
              Intersecting polygons merge into one area.
            </p>
          </div>
        </>
      )}
    </section>
  );
}
