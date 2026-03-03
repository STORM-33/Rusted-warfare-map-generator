import polygonClipping from "polygon-clipping";
import type { Polygon } from "./types";

let nextId = 1;

export function generatePolygonId(): number {
  return nextId++;
}

// Mirror a single vertex using the same logic as the Rust backend
type MirroringMode = "none" | "horizontal" | "vertical" | "diagonal1" | "diagonal2" | "both";

function mirrorVertex(
  row: number, col: number, rows: number, cols: number, mode: MirroringMode,
): [number, number][] {
  const result: [number, number][] = [];
  switch (mode) {
    case "horizontal":
      result.push([rows - 1 - row, col]);
      break;
    case "vertical":
      result.push([row, cols - 1 - col]);
      break;
    case "diagonal1":
      result.push([col, row]);
      break;
    case "diagonal2":
      result.push([rows - 1 - col, cols - 1 - row]);
      break;
    case "both":
      result.push([rows - 1 - row, col]);
      result.push([row, cols - 1 - col]);
      result.push([rows - 1 - row, cols - 1 - col]);
      break;
  }
  return result;
}

// Mirror all vertices of a polygon → produce mirrored polygon copies
export function mirrorPolygonVertices(
  polygon: Polygon, rows: number, cols: number, mode: string,
): Polygon[] {
  const m = mode as MirroringMode;
  if (m === "none") return [];

  const numMirrors = m === "both" ? 3 : 1;
  const mirroredPolygons: Polygon[] = [];

  for (let mi = 0; mi < numMirrors; mi++) {
    const newVerts: [number, number][] = polygon.vertices.map(([r, c]) => {
      const mirrors = mirrorVertex(r, c, rows, cols, m);
      return mirrors[mi] ?? [r, c];
    });
    mirroredPolygons.push({
      id: -polygon.id * 10 - (mi + 1),
      vertices: newVerts,
      edgeGaps: [...polygon.edgeGaps],
      closed: polygon.closed,
    });
  }

  return mirroredPolygons;
}

// Check if two closed polygons intersect (share any area)
export function polygonsIntersect(polyA: Polygon, polyB: Polygon): boolean {
  if (!polyA.closed || !polyB.closed) return false;
  if (polyA.vertices.length < 3 || polyB.vertices.length < 3) return false;

  const bbA = boundingBox(polyA.vertices);
  const bbB = boundingBox(polyB.vertices);
  if (bbA.maxR < bbB.minR || bbB.maxR < bbA.minR ||
      bbA.maxC < bbB.minC || bbB.maxC < bbA.minC) {
    return false;
  }

  for (const [r, c] of polyA.vertices) {
    if (isPointInPolygon(polyB.vertices, r, c)) return true;
  }
  for (const [r, c] of polyB.vertices) {
    if (isPointInPolygon(polyA.vertices, r, c)) return true;
  }
  const edgesA = getEdges(polyA.vertices);
  const edgesB = getEdges(polyB.vertices);
  for (const eA of edgesA) {
    for (const eB of edgesB) {
      if (edgesIntersect(eA, eB)) return true;
    }
  }
  return false;
}

function boundingBox(vertices: [number, number][]) {
  let minR = Infinity, maxR = -Infinity, minC = Infinity, maxC = -Infinity;
  for (const [r, c] of vertices) {
    minR = Math.min(minR, r); maxR = Math.max(maxR, r);
    minC = Math.min(minC, c); maxC = Math.max(maxC, c);
  }
  return { minR, maxR, minC, maxC };
}

function getEdges(vertices: [number, number][]): [[number, number], [number, number]][] {
  const edges: [[number, number], [number, number]][] = [];
  for (let i = 0; i < vertices.length; i++) {
    edges.push([vertices[i], vertices[(i + 1) % vertices.length]]);
  }
  return edges;
}

function cross(o: [number, number], a: [number, number], b: [number, number]): number {
  return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
}

function onSegment(p: [number, number], q: [number, number], r: [number, number]): boolean {
  return (
    Math.min(p[0], r[0]) <= q[0] && q[0] <= Math.max(p[0], r[0]) &&
    Math.min(p[1], r[1]) <= q[1] && q[1] <= Math.max(p[1], r[1])
  );
}

function edgesIntersect(
  e1: [[number, number], [number, number]],
  e2: [[number, number], [number, number]],
): boolean {
  const [p1, q1] = e1;
  const [p2, q2] = e2;
  const d1 = cross(p2, q2, p1);
  const d2 = cross(p2, q2, q1);
  const d3 = cross(p1, q1, p2);
  const d4 = cross(p1, q1, q2);
  if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
      ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) {
    return true;
  }
  if (d1 === 0 && onSegment(p2, p1, q2)) return true;
  if (d2 === 0 && onSegment(p2, q1, q2)) return true;
  if (d3 === 0 && onSegment(p1, p2, q1)) return true;
  if (d4 === 0 && onSegment(p1, q2, q1)) return true;
  return false;
}

export function isPointInPolygon(
  vertices: [number, number][], row: number, col: number,
): boolean {
  let inside = false;
  const n = vertices.length;
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const [ri, ci] = vertices[i];
    const [rj, cj] = vertices[j];
    if (ri > row !== rj > row &&
        col < ((cj - ci) * (row - ri)) / (rj - ri) + ci) {
      inside = !inside;
    }
  }
  return inside;
}

export function isPointInsideAnyPolygon(
  polygons: Polygon[], row: number, col: number,
): boolean {
  for (const poly of polygons) {
    if (!poly.closed || poly.vertices.length < 3) continue;
    if (isPointInPolygon(poly.vertices, row, col)) return true;
  }
  return false;
}

export function findPolygonAtPoint(
  polygons: Polygon[], row: number, col: number,
): Polygon | null {
  for (let i = polygons.length - 1; i >= 0; i--) {
    const poly = polygons[i];
    if (!poly.closed || poly.vertices.length < 3) continue;
    if (isPointInPolygon(poly.vertices, row, col)) return poly;
  }
  return null;
}

// Check if two closed polygons have edges that actually cross each other.
// Returns false for pure containment (one fully inside the other) — this
// allows telescopic/nested hills.
function polygonsHaveEdgeCrossing(polyA: Polygon, polyB: Polygon): boolean {
  if (!polyA.closed || !polyB.closed) return false;
  if (polyA.vertices.length < 3 || polyB.vertices.length < 3) return false;

  const bbA = boundingBox(polyA.vertices);
  const bbB = boundingBox(polyB.vertices);
  if (bbA.maxR < bbB.minR || bbB.maxR < bbA.minR ||
      bbA.maxC < bbB.minC || bbB.maxC < bbA.minC) {
    return false;
  }

  const edgesA = getEdges(polyA.vertices);
  const edgesB = getEdges(polyB.vertices);
  for (const eA of edgesA) {
    for (const eB of edgesB) {
      if (edgesIntersect(eA, eB)) return true;
    }
  }
  return false;
}

// Union a set of polygons that have crossing edges into merged polygon(s).
function unionPolygons(toMerge: Polygon[]): Polygon[] {
  // Convert [row, col] → [col, row] for polygon-clipping (expects [x, y])
  const clipPolygons = toMerge.map((p) =>
    [p.vertices.map(([r, c]): [number, number] => [c, r]).concat([[p.vertices[0][1], p.vertices[0][0]]])] as [number, number][][]
  );

  const result = polygonClipping.union(clipPolygons[0], ...clipPolygons.slice(1));

  const mergedPolygons: Polygon[] = [];
  for (const multiPoly of result) {
    const outerRing = multiPoly[0];
    if (!outerRing || outerRing.length < 4) continue;
    const vertices: [number, number][] = outerRing
      .slice(0, -1)
      .map(([x, y]): [number, number] => [Math.round(y), Math.round(x)]);
    if (vertices.length < 3) continue;
    mergedPolygons.push({
      id: generatePolygonId(),
      vertices,
      edgeGaps: vertices.map(() => false),
      closed: true,
    });
  }
  return mergedPolygons;
}

// Find all polygons transitively connected to `seed` via edge crossings.
// Returns indices into `pool` that form the connected group.
function findCrossingGroup(seed: Polygon, pool: Polygon[]): Set<number> {
  const group = new Set<number>();
  const queue = [seed];
  while (queue.length > 0) {
    const current = queue.pop()!;
    for (let i = 0; i < pool.length; i++) {
      if (group.has(i)) continue;
      if (polygonsHaveEdgeCrossing(current, pool[i])) {
        group.add(i);
        queue.push(pool[i]);
      }
    }
  }
  return group;
}

// Merge a newly closed polygon (and its mirrored copies) with any existing
// polygons whose edges actually cross. Polygons that are fully contained
// inside another (no edge crossings) are kept separate for telescopic hills.
export function mergeIntersectingPolygons(
  existing: Polygon[],
  newPoly: Polygon,
  mirrorRows?: number,
  mirrorCols?: number,
  mirrorMode?: string,
): Polygon[] {
  if (!newPoly.closed || newPoly.vertices.length < 3) {
    return [...existing, newPoly];
  }

  // Build the list of new polygons to add: the user polygon + its mirrored copies
  const newPolygons: Polygon[] = [newPoly];
  if (mirrorMode && mirrorMode !== "none" && mirrorRows && mirrorCols) {
    newPolygons.push(...mirrorPolygonVertices(newPoly, mirrorRows, mirrorCols, mirrorMode));
  }

  // Start with all existing polygons, then add new ones one-by-one,
  // merging with crossing groups each time.
  let pool = [...existing];

  for (const np of newPolygons) {
    const crossingIndices = findCrossingGroup(np, pool);

    if (crossingIndices.size === 0) {
      pool.push(np);
      continue;
    }

    const nonCrossing = pool.filter((_, i) => !crossingIndices.has(i));
    const toMerge = [...Array.from(crossingIndices).map((i) => pool[i]), np];

    try {
      const merged = unionPolygons(toMerge);
      pool = merged.length > 0
        ? [...nonCrossing, ...merged]
        : [...nonCrossing, np]; // fallback
    } catch {
      pool = [...nonCrossing, np]; // fallback
    }
  }

  return pool;
}
