# Polygon Walls Implementation Guide

## Goal

Add polygon-based wall drawing mode alongside the existing brush mode in the Hill Drawing step (step 1). Both modes coexist — brush walls and polygon walls are merged before the backend processes them.

## Architecture Overview

This is a Rust/WASM map generator. The stack:
- **Frontend**: Next.js + React (TypeScript), in `web/src/`
- **Backend**: Rust compiled to WASM, in `web/rust-map-gen/src/`
- **Communication**: Frontend sends actions to a web worker (`web/src/workers/wasm.worker.ts`) which calls `rpc_call(method, jsonParams)` on the WASM module
- **Key backend actions**: `draw_walls`, `clear_walls` (see `web/rust-map-gen/src/lib.rs:52-62`)

### Critical: How the backend `draw_walls` works

`pipeline::draw_walls()` in `web/rust-map-gen/src/pipeline.rs:82-119`:
- Takes `points`, `value` (1=wall, 2=gap), and `brush_size`
- **Applies brush expansion** via `expand_brush()` — each point is expanded to a square of `brush_size`
- **Applies mirroring** via `get_mirrored_positions()` — points are automatically mirrored based on `state.mirroring`
- **Checks coastline** — won't place walls on water tiles
- Writes to `state.wall_matrix`

**This means**: If you send pre-rasterized polygon wall cells to `draw_walls`, they will get brush-expanded AND mirrored again. You must account for this.

### How the frontend currently draws walls (brush mode)

1. User drags on canvas → `MapCanvas` fires `onDraw(points, value)` with cell coordinates
2. `WizardApp.handleDrawWalls` calls `callAction("draw_walls", { points, value, brush_size: brushSize })`
3. Backend applies brush expansion + mirroring + coastline check, updates `wall_matrix`
4. Updated snapshot (including `wall_matrix`) flows back to frontend for rendering

## Files to Modify

### 1. `web/src/lib/types.ts` — Add types
```typescript
export interface Polygon {
  id: number;
  vertices: [number, number][]; // [row, col] grid coords
  edgeGaps: boolean[];          // per-edge: true = no wall on this edge
  closed: boolean;
}

export type WallDrawingMode = "brush" | "polygon";
```

### 2. `web/src/lib/polygonUtils.ts` — NEW FILE, polygon geometry utilities
Implement:
- `rasterizeLine(r0, c0, r1, c1)` — Bresenham line → `[row, col][]`
- `getPolygonPerimeter(polygon)` — rasterize all edges (skip gap edges) → `[row, col][]` with deduplication
- `getAllPolygonWallCells(polygons)` — union of all closed polygon perimeters → `[row, col][]`
- `generatePolygonId()` — incrementing counter
- `findPolygonAtPoint(polygons, row, col)` — ray casting point-in-polygon test for selection

### 3. `web/src/components/MapCanvas.tsx` — Add polygon interaction mode and rendering

**New interaction mode**: `"polygon"` (alongside existing `"none" | "draw" | "click"`)

**New props**:
```typescript
polygons?: Polygon[];
drawingPolygon?: Polygon | null;
selectedPolygonId?: number | null;
onPolygonClick?: (row: number, col: number) => void;
onPolygonRightClick?: () => void;
```

**Polygon canvas overlay**: Add a third `<canvas>` layer for rendering polygon visuals (vertices, edges, fill). Render using a `useCallback` + `useEffect` that redraws when polygons/drawingPolygon/selectedPolygonId change.

**Cell size calculation**: The existing `eventToCell` function handles coordinate mapping correctly for both "sampled" and "full" render modes. For the polygon overlay canvas, you MUST use the same cell size logic: `activeRenderModeRef.current === "full" ? 20 : 1`. The canvas dimensions must match the base canvas (use `baseCanvasRef.current.width/height`). Do NOT recalculate cell size independently — reuse the same approach.

**Polygon mode interactions**:
- Left click → `onPolygonClick(row, col)`
- Right click → `onPolygonRightClick()`
- Pointer move → update cursor position ref for preview line (last vertex → cursor)
- Do NOT handle double-click for closing — closing is done by clicking the first vertex

**Polygon rendering** (on the polygon overlay canvas):
- Closed polygons: semi-transparent fill, solid edges (dashed for gap edges), vertex dots
- Drawing polygon: partial edges + dashed preview line from last vertex to cursor
- Selected polygon: highlighted differently (brighter color)
- Vertex positions: `col * cellSize + cellSize/2, row * cellSize + cellSize/2` (note: col=x, row=y)

### 4. `web/src/components/WizardApp.tsx` — Polygon state management and wall syncing

**New state**:
```typescript
const [wallMode, setWallMode] = useState<WallDrawingMode>("brush");
const [polygons, setPolygons] = useState<Polygon[]>([]);
const [drawingPolygon, setDrawingPolygon] = useState<Polygon | null>(null);
const [selectedPolygonId, setSelectedPolygonId] = useState<number | null>(null);
```

**CRITICAL: Wall syncing approach**

Both brush and polygon walls must coexist in the backend's single `wall_matrix`. The sync strategy:

1. **Brush mode**: works exactly as before — `handleDrawWalls` sends incremental points to `draw_walls` with the user's `brushSize`. The backend handles expansion and mirroring.

2. **Polygon mode**: When a polygon is closed/modified/deleted, we must rebuild the wall_matrix:
   - Call `clear_walls` to reset `wall_matrix` to zero
   - Re-send all polygon wall cells: call `draw_walls` with `{ points: allPolygonCells, value: 1, brush_size: 1 }`
   - **BUT**: `draw_walls` applies mirroring automatically. Polygon vertices are placed by the user at exact screen positions — they should NOT be mirrored again. **You need to handle this.**

**Options for mirroring**:
- **Option A (recommended)**: Add a new backend action `set_wall_cells` that writes directly to `wall_matrix` without brush expansion or mirroring. This is cleanest. Add it to `lib.rs`, `pipeline.rs`, types, and worker.
- **Option B**: Send polygon cells through `draw_walls` with `brush_size: 1` and accept that they'll be mirrored. This means the user only needs to draw one polygon and the mirrored copies appear automatically — which might actually be desirable UX for a mirrored map. In this case, polygon vertices themselves should be displayed with mirroring on the canvas overlay too, so the user sees what they'll get.

Pick **Option B** for now — it's simpler and arguably better UX since the whole map uses mirroring. The user draws one polygon, walls appear in all mirrored positions. The polygon overlay rendering should show mirrored polygon outlines as visual feedback.

**When switching modes**: Brush strokes are lost when polygon sync runs `clear_walls`. This is acceptable for now — document it in the UI. Alternatively, track brush strokes separately and replay them during sync, but that adds complexity.

**Actually, simpler approach**: Don't try to preserve brush strokes when using polygons. When the user switches to polygon mode, warn that brush walls will be replaced. When they switch back to brush mode, polygon walls remain in the matrix until the user clears them. This avoids complex dual-source merging.

**Polygon click handler** (`handlePolygonClick`):
- If currently drawing a polygon:
  - If click is near first vertex (within threshold ~3 cells) and ≥3 vertices → close the polygon, add to `polygons`, sync walls
  - Otherwise → add vertex to `drawingPolygon`
- If not drawing:
  - If click is inside an existing polygon → select it
  - Otherwise → start new polygon with first vertex

**Polygon right-click handler**: Cancel current drawing, or deselect.

**Polygon operations** (shown in sidebar):
- Delete selected polygon → remove from `polygons`, sync
- Toggle edge gaps on selected polygon → update `edgeGaps`, sync
- Clear all polygons → empty `polygons`, `clear_walls`
- Cancel drawing → discard `drawingPolygon`

**interactionMode calculation**: When `currentStep === 1`, use `wallMode === "polygon" ? "polygon" : "draw"`.

**Reset handling**: Clear polygon state in `handleFinish`.

### 5. `web/src/components/steps/HillDrawingStep.tsx` — Mode toggle UI

Add a mode toggle (Brush / Polygon) at the top. Show brush controls or polygon controls depending on mode.

**Brush mode UI**: exactly as current (Wall Brush, Gap Brush, Eraser, Clear, brush size slider)

**Polygon mode UI**:
- Status text: "Drawing polygon — N vertices placed. Click first vertex to close." or "Click to start a new polygon."
- Buttons: Cancel Drawing (when drawing), Toggle Edge Gap (when selected), Delete Polygon (when selected), Clear All Polygons
- Polygon count display

**New props**: `wallMode`, `polygons`, `selectedPolygonId`, `drawingPolygon`, and callbacks for mode change, delete, clear, toggle gap, cancel.

### 6. `web/src/app/globals.css` — Styles

Add:
- `.map-canvas.polygon` — crosshair cursor
- `.wall-mode-toggle` — flex container for Brush/Polygon tabs
- `.mode-tab` / `.mode-tab.active` — tab button styles matching existing dark theme
- `.polygon-info` — small text for polygon count

## Implementation Order

1. Add types to `types.ts`
2. Create `polygonUtils.ts` with geometry helpers
3. Update `HillDrawingStep.tsx` with mode toggle and polygon controls
4. Update `MapCanvas.tsx` with polygon interaction mode and overlay rendering
5. Update `WizardApp.tsx` with polygon state, handlers, and wall syncing
6. Add CSS styles

## Things to NOT do

- Do NOT modify any Rust/WASM code
- Do NOT add double-click to close polygons (use first-vertex click instead)
- Do NOT try to preserve brush walls when syncing polygon walls — it's OK to replace them
- Do NOT implement vertex dragging/editing in this pass — just drawing, closing, selecting, deleting, and gap toggling
- Do NOT add an `interior_mask` concept — the backend's flood fill handles inside/outside detection from the wall_matrix alone
