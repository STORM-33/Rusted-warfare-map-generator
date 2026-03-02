# Phase 2: Polygon-Based Wall System

## Overview
Add polygon drawing mode alongside existing brush mode for wall placement. Polygons give clean boundaries with clear inside/outside detection.

## Two Wall Drawing Modes
1. **Brush mode** (existing) — freeform painting, walls at edges of painted area
2. **Polygon mode** (new) — click vertices to define polygon, walls on perimeter, fill defines interior

User can switch between modes. Both produce the same underlying wall data for the elevation algorithm.

## Polygon Data Model

```
Polygon {
    id: u32,
    vertices: Vec<(usize, usize)>,  // ordered vertices (grid coords)
    edge_gaps: Vec<bool>,            // per-edge flag: true = no wall on this edge
    closed: bool,                    // false while still drawing
}
```

- Edge i connects vertex i to vertex (i+1) % len
- `edge_gaps[i] = true` means no wall tiles generated on that edge
- Multiple independent polygons supported

## Polygon → Wall Rasterization
1. **Fill interior**: scanline fill or flood fill from polygon edges → marks "inside" tiles
2. **Generate perimeter walls**: walk each edge, rasterize line (Bresenham), place wall tiles — skip edges marked as gaps
3. **Inside/outside**: interior tiles get elevation floor, exterior tiles get gradient falloff

## Frontend UX

### Drawing Mode
- Click to place vertices
- Visual preview of polygon edges as you go
- Click first vertex or double-click to close polygon
- Minimum 3 vertices to form valid polygon

### Editing Operations
- **Select polygon**: click inside or on edge to select
- **Move vertex**: drag existing vertex to new position
- **Add vertex**: click on edge midpoint to insert new vertex
- **Delete vertex**: select vertex + delete key (polygon must keep >= 3 vertices)
- **Toggle edge gap**: click edge to toggle wall/no-wall (visual indicator: dashed line for gap)
- **Delete polygon**: select + delete entire polygon

### Visual Feedback
- Polygon fill shown as semi-transparent overlay
- Wall edges: solid line
- Gap edges: dashed line
- Vertices: draggable handles
- Selected polygon highlighted differently from others

### Mode Switching
- Toolbar toggle between brush and polygon modes
- Both modes' results coexist on the same map
- Brush areas and polygon areas both contribute to wall_matrix

## Backend Integration

### Wall Matrix Generation
When finalizing, combine all wall sources:
1. Rasterize all polygons → perimeter walls (respecting gaps) + interior mask
2. Add brush-painted walls
3. Merge into single wall_matrix for elevation algorithm

### Elevation Algorithm Awareness
The elevation algorithm (Phase 1) needs to know "inside" vs "outside":
- Polygon interiors → apply floor (minimum height = border height)
- Brush interiors → same treatment (existing edge-detection gives inside)
- Exterior → gradient falloff

This means we may need an `interior_mask` matrix alongside `wall_matrix`.

## Mirroring
- Polygons respect the map's mirroring setting
- When user draws a polygon, mirrored copies are auto-generated
- Mirrored polygons are linked — editing one updates all copies
- Gap toggles also mirror

## Constraints
- Polygons cannot enclose water tiles (height <= 0) — vertices snap away from water, or water tiles inside are excluded
- 1-tile-wide brush strokes are ignored for inside/outside (existing behavior)
- Self-intersecting polygons: either prevent or use even-odd fill rule

## Open Questions
- Should polygons snap to a coarser grid for cleaner walls?
- How to handle polygon overlaps (union? independent?)
- Should there be a "convert brush area to polygon" tool?
- Performance: real-time preview of rasterized polygon during editing?
