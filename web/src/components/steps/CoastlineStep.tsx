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

function parseImageToCoastline(file: File): Promise<{ data: Int32Array; width: number; height: number }> {
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
      const result = new Int32Array(img.width * img.height);
      for (let i = 0; i < result.length; i++) {
        const offset = i * 4;
        const r = pixels[offset];
        const g = pixels[offset + 1];
        const b = pixels[offset + 2];
        result[i] = (r > 240 && g > 240 && b > 240) ? 1 : 0;
      }
      resolve({ data: result, width: img.width, height: img.height });
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
        const { data, width: w, height: h } = await parseImageToCoastline(file);
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
          <p>Load an image: white pixels = land, everything else = water. Map dimensions are taken from the image.</p>
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
