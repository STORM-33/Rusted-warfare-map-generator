# Phase 1: Elevation Rework

## Problems with Current Approach
1. `bias_terrain_near_walls` pushes everything toward absolute level 5 — with many walls, entire map becomes stone/snow
2. Near water, ramping from level 1 to 5 in few tiles causes artifacts (not enough space for transitions between adjacent levels)
3. Tilesets only support transitions between adjacent levels — multi-level jumps produce visual glitches

## New Algorithm: Relative Elevation

### Core Principles
- Walls refuse water levels entirely (height <= 0)
- Boost is relative to local terrain, not absolute
- Interior of walled areas preserves height variation but enforces a minimum floor
- Exterior grades down naturally via distance gradient
- Max terrain level cap (level 7)

### Computing Wall Border Height
1. For each walled area, compute **average height** of interior tiles (excluding water)
2. Wall border height = `avg_interior_height + N` (N configurable, likely 2)
3. Cap at max terrain level (7)

### Elevation Application
1. **Interior tiles** (inside walled area): raise any tile below the border height up to border height (floor). Tiles already above the floor keep their height.
2. **Exterior tiles**: BFS distance from wall boundary. Lerp from wall border height back to original terrain height over a gradient radius. Each tile uses its nearest wall's border height as target.
3. **Water refusal**: never modify tiles with height <= 0

### Transition Safety Pass
After elevation boost, enforce that every terrain level band is at least 2 tiles wide:
- For each pair of tiles distance-2 apart, if height difference > 1, raise the intermediate tile(s)
- This guarantees the smoothing pass always has room for transition tiles
- Run iteratively until stable (may need multiple passes as raising tiles can cascade)

### Edge Cases
- Walled area average is close to water level: the +N boost is small, transitions stay manageable
- Overlapping wall influence zones: each tile uses nearest wall only, no stacking
- Walls on already-high terrain: boost still applies, capped at level 7
- Thin walled areas (1-2 tiles wide interior): floor = average still works, just a small area

### Implementation Location
- Modify `bias_terrain_near_walls()` in `terrain.rs`
- Add transition safety pass (new function in `smoothing.rs` or `terrain.rs`)
- Pipeline call stays in same place in `run_height_ocean()`

### Open Questions
- Exact value of N (boost amount) — start with 2, may need to be configurable
- Gradient radius for exterior falloff — currently 12, may need adjustment
- Should the safety pass run before or after the main smoothing pass? (Likely before, so smoothing has clean input)
