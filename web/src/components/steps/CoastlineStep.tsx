"use client";

import { useCallback, useRef } from "react";
import { NumberInput } from "@/components/NumberInput";

type CoastlineMode = "grid" | "image";

type CoastlineStepProps = {
  mode: CoastlineMode;
  grid: number[][];
  height: number;
  width: number;
  mirroring: string;
  tileset: number;
  imageName: string | null;
  disabled?: boolean;
  animating?: boolean;
  animationProgress?: string;
  onToggleCell: (row: number, col: number) => void;
  onHeightChange: (value: number) => void;
  onWidthChange: (value: number) => void;
  onMirroringChange: (value: string) => void;
  onTilesetChange: (value: number) => void;
  onGenerate: () => void;
  onModeChange: (mode: CoastlineMode) => void;
  onImageLoad: (data: Int32Array, width: number, height: number, name: string) => void;
  onImageClear: () => void;
};

const MIRROR_OPTIONS = [
  "none",
  "horizontal",
  "vertical",
  "diagonal1",
  "diagonal2",
  "both",
];

type ColorClass = "water" | "land" | "hill" | "gap";

function classifyColor(avgR: number, avgG: number, avgB: number): ColorClass {
  const lum = 0.299 * avgR + 0.587 * avgG + 0.114 * avgB;
  if (lum < 30) return "water";
  if (lum > 225) return "land";
  if (avgG > 60 && avgG > avgR * 1.5 && avgG > avgB * 1.5) return "gap";
  return "hill";
}

const NEIGHBORS_4: [number, number][] = [[-1, 0], [1, 0], [0, -1], [0, 1]];

function parseImageToTerrain(file: File): Promise<{ data: Int32Array; width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        reject(new Error("Could not get canvas context"));
        return;
      }
      ctx.drawImage(img, 0, 0);
      const imageData = ctx.getImageData(0, 0, img.width, img.height);
      const pixels = imageData.data;
      const W = img.width;
      const H = img.height;
      const N = W * H;

      // Step 1: Quantize colors and track per-bucket RGB sums
      const colorGrid = new Uint16Array(N);
      const bucketRSum = new Map<number, number>();
      const bucketGSum = new Map<number, number>();
      const bucketBSum = new Map<number, number>();
      const bucketCount = new Map<number, number>();

      for (let i = 0; i < N; i++) {
        const off = i * 4;
        const r = pixels[off], g = pixels[off + 1], b = pixels[off + 2];
        const key = ((r >> 5) << 10) | ((g >> 5) << 5) | (b >> 5);
        colorGrid[i] = key;
        bucketRSum.set(key, (bucketRSum.get(key) ?? 0) + r);
        bucketGSum.set(key, (bucketGSum.get(key) ?? 0) + g);
        bucketBSum.set(key, (bucketBSum.get(key) ?? 0) + b);
        bucketCount.set(key, (bucketCount.get(key) ?? 0) + 1);
      }

      // Step 2: Classify each quantized color bucket
      const bucketClass = new Map<number, ColorClass>();
      for (const [key, count] of bucketCount) {
        const avgR = (bucketRSum.get(key) ?? 0) / count;
        const avgG = (bucketGSum.get(key) ?? 0) / count;
        const avgB = (bucketBSum.get(key) ?? 0) / count;
        bucketClass.set(key, classifyColor(avgR, avgG, avgB));
      }

      // Step 3: Track which pixels are originally gap
      const isGapPixel = new Uint8Array(N);
      for (let i = 0; i < N; i++) {
        if (bucketClass.get(colorGrid[i]) === "gap") {
          isGapPixel[i] = 1;
        }
      }

      // Step 4: Noise filter — merge small connected components
      // Skip gap pixels so they aren't absorbed into surrounding zones.
      const visited = new Uint8Array(N);
      const MIN_REGION = 16;

      for (let startIdx = 0; startIdx < N; startIdx++) {
        if (visited[startIdx] || isGapPixel[startIdx]) continue;
        const thisColor = colorGrid[startIdx];
        const component: number[] = [];
        const neighborColors = new Map<number, number>();
        const queue = [startIdx];
        visited[startIdx] = 1;

        while (queue.length > 0) {
          const idx = queue.pop()!;
          component.push(idx);
          const row = (idx / W) | 0, col = idx % W;
          for (const [dr, dc] of NEIGHBORS_4) {
            const nr = row + dr, nc = col + dc;
            if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
            const ni = nr * W + nc;
            if (isGapPixel[ni]) continue;
            if (visited[ni]) {
              if (colorGrid[ni] !== thisColor) {
                neighborColors.set(colorGrid[ni], (neighborColors.get(colorGrid[ni]) ?? 0) + 1);
              }
              continue;
            }
            if (colorGrid[ni] === thisColor) {
              visited[ni] = 1;
              queue.push(ni);
            } else {
              neighborColors.set(colorGrid[ni], (neighborColors.get(colorGrid[ni]) ?? 0) + 1);
            }
          }
        }

        if (component.length < MIN_REGION && neighborColors.size > 0) {
          let bestNeighbor = thisColor, bestCount = 0;
          for (const [nk, cnt] of neighborColors) {
            if (cnt > bestCount) { bestCount = cnt; bestNeighbor = nk; }
          }
          for (const idx of component) {
            colorGrid[idx] = bestNeighbor;
          }
        }
      }

      // Step 5: BFS depth assignment from land pixels.
      // Gap pixels are skipped during BFS — their depth is fixed afterward.
      const result = new Int32Array(N);
      const depthAssigned = new Int8Array(N).fill(-1);
      const bfsQueue: number[] = [];
      const bfsDepth: number[] = [];

      for (let i = 0; i < N; i++) {
        if (isGapPixel[i]) continue;
        const cls = bucketClass.get(colorGrid[i]) ?? "water";
        if (cls === "water") {
          result[i] = 0;
          depthAssigned[i] = 0;
        } else if (cls === "land") {
          result[i] = 1;
          depthAssigned[i] = 0;
          bfsQueue.push(i);
          bfsDepth.push(0);
        }
      }

      let head = 0;
      while (head < bfsQueue.length) {
        const idx = bfsQueue[head];
        const d = bfsDepth[head];
        head++;
        const row = (idx / W) | 0, col = idx % W;
        const myColor = colorGrid[idx];

        for (const [dr, dc] of NEIGHBORS_4) {
          const nr = row + dr, nc = col + dc;
          if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
          const ni = nr * W + nc;
          if (depthAssigned[ni] >= 0 || isGapPixel[ni]) continue;

          const neighborCls = bucketClass.get(colorGrid[ni]) ?? "water";
          if (neighborCls === "water") continue;

          const newDepth = Math.min(colorGrid[ni] !== myColor ? d + 1 : d, 9);
          depthAssigned[ni] = newDepth;
          result[ni] = newDepth + 1;
          bfsQueue.push(ni);
          bfsDepth.push(newDepth);
        }
      }

      // Unvisited non-gap hill pixels (disconnected from land) → flat land
      for (let i = 0; i < N; i++) {
        if (!isGapPixel[i] && depthAssigned[i] < 0) {
          result[i] = 1;
        }
      }

      // Step 6: Fix gap pixel depths by propagating max depth from non-gap
      // neighbors. Gap pixels should match the zone they sit on so they
      // don't create false boundaries on neighboring non-gap cells.
      for (let changed = true; changed; ) {
        changed = false;
        for (let i = 0; i < N; i++) {
          if (!isGapPixel[i]) continue;
          const r = (i / W) | 0, c = i % W;
          let maxVal = result[i];
          for (const [dr, dc] of NEIGHBORS_4) {
            const nr = r + dr, nc = c + dc;
            if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
            const ni = nr * W + nc;
            const v = Math.abs(result[ni]);
            if (v > maxVal) maxVal = v;
          }
          if (maxVal > result[i]) {
            result[i] = maxVal;
            changed = true;
          }
        }
      }

      // Step 7: Encode gap pixels as negative values
      for (let i = 0; i < N; i++) {
        if (isGapPixel[i] && result[i] > 0) {
          result[i] = -result[i];
        }
      }

      resolve({ data: result, width: W, height: H });
    };
    img.onerror = () => reject(new Error("Failed to load image"));
    img.src = URL.createObjectURL(file);
  });
}

export function CoastlineStep({
  mode,
  grid,
  height,
  width,
  mirroring,
  tileset,
  imageName,
  disabled,
  animating,
  animationProgress,
  onToggleCell,
  onHeightChange,
  onWidthChange,
  onMirroringChange,
  onTilesetChange,
  onGenerate,
  onModeChange,
  onImageLoad,
  onImageClear,
}: CoastlineStepProps) {
  const isDisabled = disabled || animating;
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;
      try {
        const { data, width: w, height: h } = await parseImageToTerrain(file);
        onImageLoad(data, w, h, file.name);
      } catch {
        // reset input so user can retry
        if (fileInputRef.current) fileInputRef.current.value = "";
      }
    },
    [onImageLoad],
  );

  return (
    <section className="panel-section">
      <h2>1. Coastline</h2>

      <div className="coast-mode-tabs">
        <button
          type="button"
          className={`mode-tab ${mode === "grid" ? "active" : ""}`}
          onClick={() => onModeChange("grid")}
          disabled={isDisabled}
        >
          Grid
        </button>
        <button
          type="button"
          className={`mode-tab ${mode === "image" ? "active" : ""}`}
          onClick={() => onModeChange("image")}
          disabled={isDisabled}
        >
          Image
        </button>
      </div>

      {mode === "grid" ? (
        <>
          <p>Toggle the 5x5 seed grid, then generate mirrored coastline terrain.</p>
          <div className="coast-grid">
            {grid.map((row, rowIndex) =>
              row.map((value, colIndex) => (
                <button
                  key={`${rowIndex}-${colIndex}`}
                  type="button"
                  className={`grid-cell ${value ? "on" : "off"}`}
                  onClick={() => onToggleCell(rowIndex, colIndex)}
                  disabled={isDisabled}
                />
              )),
            )}
          </div>
          <div className="control-grid">
            <label>
              Height
              <NumberInput value={height} min={40} max={640} step={20} onChange={onHeightChange} disabled={isDisabled} />
            </label>
            <label>
              Width
              <NumberInput value={width} min={40} max={640} step={20} onChange={onWidthChange} disabled={isDisabled} />
            </label>
            <label>
              Mirroring
              <select
                value={mirroring}
                onChange={(event) => onMirroringChange(event.target.value)}
                disabled={isDisabled}
              >
                {MIRROR_OPTIONS.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Tileset
              <select
                value={tileset}
                onChange={(event) => onTilesetChange(Number(event.target.value))}
                disabled={isDisabled}
              >
                {[
                  { id: 5, name: "Jungle" },
                  { id: 2, name: "Winter" },
                  { id: 3, name: "Volcanic" },
                  { id: 4, name: "Desert" },
                  { id: 1, name: "Forest" },
                ].map(
                  ({ id, name }) => (
                    <option key={name} value={id}>
                      {name}
                    </option>
                  ),
                )}
              </select>
            </label>
          </div>
          <button type="button" className="primary-btn" onClick={onGenerate} disabled={isDisabled}>
            {animating ? `Generating... ${animationProgress ?? ""}` : "Generate Coastline"}
          </button>
        </>
      ) : (
        <>
          <p>Load an image: black = water, white = flat land, other colors = hill zones (nested colors create higher walls). Green = gap/break in walls. Map dimensions are taken from the image.</p>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            disabled={isDisabled}
            className="image-file-input"
          />
          {imageName && (
            <div className="image-info">
              <span className="image-name">{imageName}</span>
              <span className="image-dims">{width} x {height}</span>
              <button type="button" className="ghost-btn" onClick={() => { onImageClear(); if (fileInputRef.current) fileInputRef.current.value = ""; }} disabled={isDisabled}>
                Clear
              </button>
            </div>
          )}
          <div className="control-grid">
            <label>
              Tileset
              <select
                value={tileset}
                onChange={(event) => onTilesetChange(Number(event.target.value))}
                disabled={isDisabled}
              >
                {[
                  { id: 5, name: "Jungle" },
                  { id: 2, name: "Winter" },
                  { id: 3, name: "Volcanic" },
                  { id: 4, name: "Desert" },
                  { id: 1, name: "Forest" },
                ].map(
                  ({ id, name }) => (
                    <option key={name} value={id}>
                      {name}
                    </option>
                  ),
                )}
              </select>
            </label>
          </div>
        </>
      )}
    </section>
  );
}
